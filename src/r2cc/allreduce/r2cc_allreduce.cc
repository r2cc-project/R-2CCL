#include "r2cc_allreduce.h"
#include "comm.h"
#include "bootstrap.h"
#include "transport.h"
#include "r2cc/oob/oob_udp.h"
#include <cctype>
#include <climits>

static constexpr int kMaxChunks = 16;
static constexpr int kEvents = 2*kMaxChunks+4;
struct r2ccAllReduceState {
  ncclComm_t healthy, aPlusH;
  int hRank, hSubRank;
  struct Parameters {
    uint64_t minBytes, failedDevs;
    int chunks, schedule, failedNode, totalDevs, explicitHca;
  } params;
  int paramsReady;
  cudaStream_t sSub, sRed, sBc;
  cudaEvent_t ev[kEvents];
  int state; // 0 uninitialized, 1 ready, 2 permanently downgraded
};

static int envInt(const char* name, int fallback) {
  const char* value = getenv(name);
  return value ? atoi(value) : fallback;
}

static int localFailedNode(struct ncclComm* comm) {
  return envInt("R2CC_FAILED_NODE", comm->r2ccRepairNode ? comm->r2ccRepairNode-1 : 0);
}

// Match the existing R2CC_DISCONNECTED_HCA parser: names, indices and numeric
// HCA suffixes, with comma/space/tab separators.
static bool matchHca(const char* list, const char* name, int dev) {
  while (*list) {
    list += strspn(list, " ,\t");
    size_t len = strcspn(list, " ,\t");
    if (!len) break;
    if (strlen(name) == len && !strncmp(list, name, len)) return true;
    unsigned value = 0;
    bool numeric = true;
    for (size_t i = 0; i < len; i++) {
      if (!isdigit((unsigned char)list[i]) || value > (INT_MAX-9u)/10u) { numeric = false; break; }
      value = value*10 + list[i]-'0';
    }
    const char* suffix = strrchr(name, '_');
    if (numeric && ((int)value == dev || (suffix && suffix[1] && atoi(suffix+1) == (int)value))) return true;
    list += len;
  }
  return false;
}

// Resolve environment settings once, before either masks or eligibility can branch.
static ncclResult_t agreeParameters(struct ncclComm* comm) {
  if (!comm->r2ccAllReduce) NCCLCHECK(ncclCalloc(&comm->r2ccAllReduce, 1));
  r2ccAllReduceState* st = comm->r2ccAllReduce;
  if (st->paramsReady) return ncclSuccess;
  r2ccAllReduceState::Parameters local = {};
  local.chunks = std::max(1, std::min(kMaxChunks, envInt("R2CC_AR_STAGE2_CHUNKS", 4)));
  // Serial Stage 1 avoids main/tail contention on the two-node PCIe testbed.
  local.schedule = std::max(0, std::min(3, envInt("R2CC_AR_SCHEDULE", 2)));
  const char* minimum = getenv("R2CC_AR_MIN_BYTES");
  local.minBytes = minimum ? strtoull(minimum, NULL, 0) : 16777216;
  local.failedNode = localFailedNode(comm);
  NCCLCHECK(comm->ncclNet->devices(&local.totalDevs));
  const char* list = getenv("R2CC_FAILED_HCA");
  local.explicitHca = list && *list;
  int last = envInt("R2CC_FAILED_NIC_COUNT", envInt("FAILURE2_SAME_SERVER", 0) == 1 ? 2 : 1);
  // The topology mask uses a uint64_t netDev set.
  if (local.totalDevs > 64) return ncclInvalidUsage;
  for (int d = 0; d < local.totalDevs; d++) {
    bool match;
    if (local.explicitHca) {
      ncclNetProperties_t props;
      NCCLCHECK(comm->ncclNet->getProperties(d, &props));
      match = matchHca(list, props.name, d);
    } else {
      match = d >= std::max(0, local.totalDevs-last);
    }
    if (match) local.failedDevs |= 1ull << d;
  }
  r2ccAllReduceState::Parameters* all = NULL;
  NCCLCHECK(ncclCalloc(&all, comm->nRanks));
  all[comm->rank] = local;
  ncclResult_t ret = bootstrapAllGather(comm->bootstrap, all, sizeof(*all));
  if (ret == ncclSuccess) {
    // One warning per differing parameter per communicator, emitted by rank 0.
    if (comm->rank == 0) {
#define R2CC_WARN_PARAM(field, name) do { \
      for (int r = 1; r < comm->nRanks; r++) if (all[r].field != all[0].field) { \
        WARN("R2CC parameter %s differs across ranks; adopting rank 0 value", name); break; \
      } \
    } while (0)
      R2CC_WARN_PARAM(chunks, "R2CC_AR_STAGE2_CHUNKS");
      R2CC_WARN_PARAM(minBytes, "R2CC_AR_MIN_BYTES");
      R2CC_WARN_PARAM(failedNode, "R2CC_FAILED_NODE");
      R2CC_WARN_PARAM(failedDevs, "R2CC_FAILED_HCA/R2CC_FAILED_NIC_COUNT/FAILURE2_SAME_SERVER");
      R2CC_WARN_PARAM(explicitHca, "R2CC_FAILED_HCA");
      R2CC_WARN_PARAM(schedule, "R2CC_AR_SCHEDULE");
      R2CC_WARN_PARAM(totalDevs, "netDev count");
#undef R2CC_WARN_PARAM
    }
    st->params = all[0];
    st->paramsReady = 1;
  }
  free(all);
  return ret;
}

static ncclResult_t computeFailedChannelMask(struct ncclComm* comm, int failedNode, bool onFailedNode, uint64_t failedDevs) {
  // Runtime connection setup must precede reading the transport's actual NICs.
  if (comm->runtimeConn && !comm->initAlgoChannels[NCCL_ALGO_RING]) {
    NCCLCHECK(ncclTransportRingConnect(comm));
    comm->initAlgoChannels[NCCL_ALGO_RING] = true;
  }
  uint64_t* masks = NULL;
  NCCLCHECK(ncclCalloc(&masks, comm->nRanks));
  ncclResult_t ret = ncclSuccess;
  if (onFailedNode) {
    for (int c = 0; c < comm->nChannels; c++) {
      for (int d : {comm->r2ccChanSendDev[c], comm->r2ccChanRecvDev[c]}) {
        if (d >= 0 && d < 64 && (failedDevs & (1ull << d))) masks[comm->rank] |= 1ull << c;
      }
    }
  }
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, masks, sizeof(uint64_t)), ret, exit);
  comm->r2ccFailedChanMask = 0;
  for (int r = 0; r < comm->nRanks; r++) comm->r2ccFailedChanMask |= masks[r];
  comm->r2ccMaskReady = 1;
  comm->r2ccMaskNode = failedNode;
  INFO(NCCL_R2CC, "R2CC mask comm=%p rank=%d failedNode=%d mask=0x%lx", comm, comm->rank, failedNode, (unsigned long)comm->r2ccFailedChanMask);
exit:
  free(masks);
  return ret;
}

ncclResult_t r2ccPrepareBalance(struct ncclComm* comm, int mode) {
  if (comm->r2ccIsChild) return ncclSuccess;
  comm->r2ccBalanceEnabled = mode >= 2;
  if (!comm->r2ccBalanceEnabled) return ncclSuccess;
  NCCLCHECK(agreeParameters(comm));
  auto& params = comm->r2ccAllReduce->params;
  int node = params.failedNode;
  if (comm->r2ccRepairNode && !comm->r2ccRepairDevs) {
    // An observed failure supersedes the simulated "last devices" default.
    // Translate the parent's failed channels back to NICs before mapping children.
    uint64_t* devs = NULL;
    NCCLCHECK(ncclCalloc(&devs, comm->nRanks));
    uint64_t observed = OobNet::Get().GlobalFailedChannelMask();
    if (comm->node == node) {
      for (int c = 0; c < comm->nChannels; c++) {
        if (!(observed & (1ull << c))) continue;
        for (int d : {comm->r2ccChanSendDev[c], comm->r2ccChanRecvDev[c]}) {
          if (d >= 0 && d < 64) devs[comm->rank] |= 1ull << d;
        }
      }
    }
    ncclResult_t ret = bootstrapAllGather(comm->bootstrap, devs, sizeof(uint64_t));
    if (ret == ncclSuccess) for (int r = 0; r < comm->nRanks; r++) comm->r2ccRepairDevs |= devs[r];
    free(devs);
    NCCLCHECK(ret);
  }
  if (comm->r2ccRepairDevs && !params.explicitHca) params.failedDevs = comm->r2ccRepairDevs;
  if (!comm->r2ccMaskReady || comm->r2ccMaskNode != node) {
    NCCLCHECK(computeFailedChannelMask(comm, node, comm->node == node, params.failedDevs));
  }
  if (!comm->r2ccIsChild) comm->r2ccFailedChanMask |= OobNet::Get().GlobalFailedChannelMask();
  return ncclSuccess;
}

#include "enqueue.h"
#include "group.h"

static void fallback(struct ncclComm* comm, int reason, const char* text) {
  unsigned bit = 1u << reason;
  if (!(comm->r2ccFallbackWarnings & bit)) {
    comm->r2ccFallbackWarnings |= bit;
    WARN("R2CC AllReduce falling back to Balance: %s", text);
  }
}

ncclResult_t r2ccAllReduceDestroy(struct ncclComm* comm, bool abort) {
  r2ccAllReduceState* st = comm->r2ccAllReduce;
  if (!st) return ncclSuccess;
  int oldDev;
  CUDACHECK(cudaGetDevice(&oldDev));
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  ncclResult_t ret = ncclSuccess, err;
  // Signal both children before reclaiming either on abort. A healthy helper may
  // still be waiting for its other communicator's work.
  if (abort) {
    for (ncclComm_t child : {st->healthy, st->aPlusH}) {
      if (child) {
        __atomic_store_n(child->abortFlag, 1, __ATOMIC_RELEASE);
        __atomic_store_n(child->abortFlagDev, 1, __ATOMIC_RELEASE);
      }
    }
  } else {
    for (cudaStream_t s : {st->sSub, st->sRed, st->sBc}) {
      if (s && cudaStreamSynchronize(s) != cudaSuccess) ret = ncclUnhandledCudaError;
    }
  }
  for (ncclComm_t child : {st->healthy, st->aPlusH}) {
    if (child) {
      err = abort ? ncclCommAbort(child) : ncclCommDestroy(child);
      if (err != ncclSuccess) ret = err;
    }
  }
  for (cudaEvent_t e : st->ev) if (e && cudaEventDestroy(e) != cudaSuccess) ret = ncclUnhandledCudaError;
  for (cudaStream_t s : {st->sSub, st->sRed, st->sBc}) if (s && cudaStreamDestroy(s) != cudaSuccess) ret = ncclUnhandledCudaError;
  free(st);
  comm->r2ccAllReduce = NULL;
  if (cudaSetDevice(oldDev) != cudaSuccess) ret = ncclUnhandledCudaError;
  return ret;
}

// All ranks agree on setup failure before choosing the fallback sequence.
static ncclResult_t agreeSetup(struct ncclComm* comm, ncclResult_t local, bool* ok) {
  int* errors = NULL;
  NCCLCHECK(ncclCalloc(&errors, comm->nRanks));
  errors[comm->rank] = local != ncclSuccess;
  ncclResult_t ret = bootstrapAllGather(comm->bootstrap, errors, sizeof(int));
  *ok = ret == ncclSuccess;
  for (int r = 0; r < comm->nRanks; r++) if (errors[r]) *ok = false;
  free(errors);
  return ret;
}

static ncclResult_t createStreams(r2ccAllReduceState* st) {
  CUDACHECK(cudaStreamCreateWithFlags(&st->sSub, cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&st->sRed, cudaStreamNonBlocking));
  CUDACHECK(cudaStreamCreateWithFlags(&st->sBc, cudaStreamNonBlocking));
  for (int i = 0; i < kEvents; i++) CUDACHECK(cudaEventCreateWithFlags(&st->ev[i], cudaEventDisableTiming));
  return ncclSuccess;
}

static ncclResult_t prepare(struct ncclComm* comm, int node, int helper) {
  NCCLCHECK(agreeParameters(comm));
  r2ccAllReduceState* st = comm->r2ccAllReduce;
  if (st->state) return ncclSuccess;
  st->hRank = helper;
  st->hSubRank = 0;
  for (int r = 0; r < helper; r++) if (comm->rankToNode[r] == node) st->hSubRank++;
  bool ok;
  ncclResult_t ret = createStreams(st);
  NCCLCHECK(agreeSetup(comm, ret, &ok));
  if (ok) {
    ret = ncclCommSplit(comm, comm->node == node ? NCCL_SPLIT_NOCOLOR : 0, comm->rank, &st->healthy, NULL);
    NCCLCHECK(agreeSetup(comm, ret, &ok));
  }
  if (ok) {
    ret = ncclCommSplit(comm, comm->node == node || comm->rank == helper ? 0 : NCCL_SPLIT_NOCOLOR,
                       comm->rank, &st->aPlusH, NULL);
    NCCLCHECK(agreeSetup(comm, ret, &ok));
  }
  if (ok) {
    if (st->aPlusH) {
      st->aPlusH->r2ccBalanceEnabled = 1;
      st->aPlusH->r2ccRepairDevs = comm->r2ccRepairDevs;
      // Node numbering in a split child need not match the parent.
      ret = computeFailedChannelMask(st->aPlusH, node, comm->node == node, st->params.failedDevs);
    }
    NCCLCHECK(agreeSetup(comm, ret, &ok));
  }
  st->state = ok ? 1 : 2;
  return ncclSuccess;
}

static ncclResult_t enqueue(ncclFunc_t coll, const void* send, void* recv, size_t count,
    ncclDataType_t type, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  ncclInfo info = {coll, ncclFuncToString(coll), send, recv, count, type, op, root, comm, stream,
    coll == ncclFuncAllReduce ? ALLREDUCE_CHUNKSTEPS : 1,
    coll == ncclFuncAllReduce ? ALLREDUCE_SLICESTEPS : 1};
  return ncclEnqueueCheck(&info);
}

static ncclResult_t run(struct ncclInfo* info) {
  ncclComm_t comm = info->comm;
  NCCLCHECK(agreeParameters(comm));
  r2ccAllReduceState* st = comm->r2ccAllReduce;
  const auto& params = st->params;
  int node = params.failedNode, helper = -1, aRanks = 0;
  for (int r = 0; r < comm->nRanks; r++) {
    if (comm->rankToNode[r] == node) aRanks++;
    else if (helper == -1) helper = r;
  }
  int es = ncclTypeSize(info->datatype);
  size_t minBytes = params.minBytes;
  int total = params.totalDevs, failed = __builtin_popcountll(params.failedDevs);
  size_t tail = 0, main = info->count;
  int reason = -1;
  const char* why = NULL;
  cudaStreamCaptureStatus capture;
  CUDACHECK(cudaStreamIsCapturing(info->stream, &capture));
  if (comm->nNodes < 2 || !aRanks || helper < 0) { reason = 0; why = "no failed/healthy node partition"; }
  else if (es <= 0 || info->count < minBytes/(size_t)es + (minBytes%(size_t)es != 0)) { reason = 1; why = "message below minimum size"; }
  else if (info->op != ncclSum && info->op != ncclProd && info->op != ncclMin && info->op != ncclMax) { reason = 2; why = "average or user reduction"; }
  else if (ncclGroupDepth) { reason = 3; why = "user group"; }
  else if (capture != cudaStreamCaptureStatusNone) { reason = 4; why = "stream capture"; }
  else if (!comm->config.blocking) { reason = 5; why = "nonblocking communicator"; }
  if (reason < 0) {
    if (total > 0) tail = (info->count / total)*failed + ((info->count % total)*failed)/total;
    tail -= tail % std::max(1, 16/es);
    main -= tail;
    if (!main || !tail) { reason = 6; why = "empty main or tail"; }
  }
  if (reason < 0) {
    NCCLCHECK(prepare(comm, node, helper));
    if (comm->r2ccAllReduce->state != 1) { reason = 7; why = "subcommunicator setup failed"; }
  }
  int k = std::min<size_t>(params.chunks, tail);
  INFO(NCCL_R2CC, "R2CC AllReduce count=%zu mainCount=%zu tailCount=%zu K=%d failedNode=%d h=%d X=%.6f schedule=%d fallback=%d",
       info->count, main, tail, k, node, helper, total ? (double)failed/total : 0., params.schedule, reason);
  if (reason >= 0) {
    fallback(comm, reason, why);
    return ncclEnqueueCheck(info);
  }
  const char* send = (const char*)info->sendbuff;
  char* recv = (char*)info->recvbuff;
  cudaStream_t s = info->stream;
  CUDACHECK(cudaEventRecord(st->ev[0], s));
  NCCLCHECK(enqueue(ncclFuncAllReduce, send, recv, main, info->datatype, info->op, 0, comm, s));
  // This event precedes the user's final join, so waiting on it cannot create
  // a cycle through the Reduce/Broadcast pipeline.
  if (params.schedule >= 1) {
    CUDACHECK(cudaEventRecord(st->ev[4+kMaxChunks], s));
    CUDACHECK(cudaStreamWaitEvent(st->sRed, st->ev[4+kMaxChunks], 0));
    CUDACHECK(cudaStreamWaitEvent(st->sBc, st->ev[4+kMaxChunks], 0));
  }
  CUDACHECK(cudaStreamWaitEvent(st->sSub, st->ev[params.schedule >= 2 ? 4+kMaxChunks : 0], 0));
  if (st->healthy) NCCLCHECK(enqueue(ncclFuncAllReduce, send+main*es, recv+main*es, tail,
                                   info->datatype, info->op, 0, st->healthy, st->sSub));
  CUDACHECK(cudaEventRecord(st->ev[1], st->sSub));
  CUDACHECK(cudaStreamWaitEvent(st->sRed, st->ev[1], 0));
  CUDACHECK(cudaStreamWaitEvent(st->sBc, st->ev[0], 0));
  // Healthy non-helper ranks must finish producing P before broadcast overwrites it.
  CUDACHECK(cudaStreamWaitEvent(st->sBc, st->ev[1], 0));
  for (int i = 0; i < k; i++) {
    size_t lo = (size_t)i*tail/k, hi = (size_t)(i+1)*tail/k;
    size_t offset = (main+lo)*es;
    if (st->aPlusH) NCCLCHECK(enqueue(ncclFuncReduce, comm->rank == helper ? recv+offset : send+offset,
        recv+offset, hi-lo, info->datatype, info->op, st->hSubRank, st->aPlusH, st->sRed));
    CUDACHECK(cudaEventRecord(st->ev[4+i], st->sRed));
    if (params.schedule == 3) {
      CUDACHECK(cudaStreamWaitEvent(st->sBc, st->ev[4+i], 0));
      NCCLCHECK(enqueue(ncclFuncBroadcast, recv+offset, recv+offset, hi-lo,
          info->datatype, ncclSum, helper, comm, st->sBc));
      // Each wait captures this record before the event is reused. Broadcast_i
      // waits Reduce_i, and only Reduce_{i+1} waits Broadcast_i: no cycle.
      CUDACHECK(cudaEventRecord(st->ev[2], st->sBc));
      if (i+1 < k) CUDACHECK(cudaStreamWaitEvent(st->sRed, st->ev[2], 0));
    }
  }
  for (int i = 0; params.schedule != 3 && i < k; i++) {
    size_t lo = (size_t)i*tail/k, hi = (size_t)(i+1)*tail/k;
    char* ptr = recv+(main+lo)*es;
    CUDACHECK(cudaStreamWaitEvent(st->sBc, st->ev[4+i], 0));
    NCCLCHECK(enqueue(ncclFuncBroadcast, ptr, ptr, hi-lo, info->datatype, ncclSum, helper, comm, st->sBc));
  }
  CUDACHECK(cudaEventRecord(st->ev[2], st->sBc));
  CUDACHECK(cudaEventRecord(st->ev[3], st->sRed));
  CUDACHECK(cudaStreamWaitEvent(s, st->ev[2], 0));
  CUDACHECK(cudaStreamWaitEvent(s, st->ev[3], 0));
  return ncclSuccess;
}

ncclResult_t r2ccAllReduce(struct ncclInfo* info) {
  int oldDev;
  CUDACHECK(cudaGetDevice(&oldDev));
  CUDACHECK(cudaSetDevice(info->comm->cudaDev));
  ncclResult_t ret = run(info);
  CUDACHECK(cudaSetDevice(oldDev));
  return ret;
}
