/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "net.h"
#include "graph.h"
#include "proxy.h"
#include "collectives.h"
#include "gdrwrap.h"
#include "shmutils.h"
#include "p2p.h"
#include "profiler.h"
#include <time.h>
#include <inttypes.h>
#include "transport.h"
#include "shm.h"
#include "r2cc/oob/oob_udp.h"
#include <cctype>

static_assert(sizeof(ncclNetHandle_t) <= CONNECT_SIZE, "NET Connect info is too large");

#define NCCL_NET_MAP_HOSTMEM 0
#define NCCL_NET_MAP_DEVMEM 1
#define NCCL_NET_MAP_SHARED_HOSTMEM 2
#define NCCL_NET_MAP_SHARED_DEVMEM 3
#define NCCL_NET_MAP_GDCMEM 4
#define NCCL_NET_MAP_MEMS 5

#define NCCL_NET_MAP_MASK_DEVMEM 0x40000000
#define NCCL_NET_MAP_MASK_SHARED 0x80000000
#define NCCL_NET_MAP_MASK_USED   0x20000000
#define NCCL_NET_MAP_MASK_OFFSET 0x1fffffff

#define NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName) \
  ((mapStruct)->offsets.offsetName >> 30)

#define NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName >> 29) == 0)

#define NCCL_NET_MAP_GET_POINTER(mapStruct, cpuOrGpu, offsetName) \
  (NCCL_NET_MAP_OFFSET_NULL(mapStruct, offsetName) ? NULL : \
   (mapStruct)->mems[NCCL_NET_MAP_OFFSET_BANK(mapStruct, offsetName)].cpuOrGpu##Ptr + ((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_OFFSET))

#define NCCL_NET_MAP_DEV_MEM(mapStruct, offsetName) \
  (((mapStruct)->offsets.offsetName & NCCL_NET_MAP_MASK_DEVMEM) != 0)

#define NCCL_NET_MAP_ADD_POINTER(mapStruct, shared, dev, memSize, offsetName) do { \
    int bank = NCCL_NET_MAP_MASK_USED + (dev)*NCCL_NET_MAP_MASK_DEVMEM + (shared)*NCCL_NET_MAP_MASK_SHARED; \
    if ((shared) == 0) { \
      if (dev) { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_DEVMEM].size += memSize; \
      } else { \
        (mapStruct)->offsets.offsetName = bank + (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size; \
        (mapStruct)->mems[NCCL_NET_MAP_HOSTMEM].size += memSize; \
      } \
    } else { \
      (mapStruct)->offsets.offsetName = bank; \
    } \
} while (0);

struct connectMapMem{
  char* gpuPtr;
  char* cpuPtr;
  int size;
  ncclIpcDesc ipcDesc;
  ncclShmIpcDesc_t attachDesc;
  ncclShmIpcDesc_t createDesc;
};

struct connectMap {
  int sameProcess;
  int shared;
  int cudaDev;
  // First 3 bits of offsets determine the mem bank. 001 is host mem, 011 is dev mem, 101 is shared host mem and 111 is shared dev mem.
  struct connectMapMem mems[NCCL_NET_MAP_MEMS];
  // Offsets. 3 MSBs indicate mem bank, 111 indicates NULL.
  struct {
    uint32_t sendMem;
    uint32_t recvMem;
    uint32_t buffs[NCCL_NUM_PROTOCOLS];
  } offsets;
};


struct sendNetResources {
  struct connectMap map;
  void* netSendComm;
  

  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;


  int netDev;
  int useGdr;
  int useDmaBuf;
  int maxRecvs;
  int netDeviceVersion;
  ncclNetDeviceType netDeviceType;
  ncclNetDeviceHandle_t* netDeviceHandle;
  void* mhandles[NCCL_NUM_PROTOCOLS];


  uint64_t* gdcSync;
  void* gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  
  // R2CC: Track connection log state
  int primaryConnStartLogged;
  int backupConnStartLogged;
  int connectInitLogged;
  uint64_t llLastCleaning;


  void *netSendCommBackup;
  void *mhandlesBackup[NCCL_NUM_PROTOCOLS];
  int netDevBackup;
  int useGdrBackup;
  int useDmaBufBackup;
  int maxRecvsBackup;
  int netDeviceVersionBackup;
  ncclNetDeviceType netDeviceTypeBackup;
  ncclNetDeviceHandle_t *netDeviceHandleBackup;

  int useBackup;
  int forceBackup;
  int forceBackupNotified;
  int stepSyncRequested;
  int stepSyncWaitIters;
  uint64_t failoverEpoch;
  int failoverWaitAck;
  uint64_t failoverReqAbsStep;
  uint64_t failoverWaitStartMs;
  uint64_t failoverWaitLastWarnMs;
  int stallReason;
  uint64_t stallStartMs;
  uint64_t stallLastWarnMs;
  uint64_t stallPosted;
  uint64_t stallTransmitted;
  uint64_t stallDone;
  int state;
};

struct recvNetResources {
  struct connectMap map;
  void* netListenComm;
  void* netRecvComm;
  struct ncclSendMem* sendMem;
  struct ncclRecvMem* recvMem;

  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int tpRemoteProxyRank;
  int netDev;
  int useGdr;
  int useDmaBuf;
  int needFlush;
  int maxRecvs;
  uint64_t* gdcSync;
  uint64_t* gdcFlush;
  void* gdrDesc;
  int shared;
  int channelId;
  int connIndex;
  char* buffers[NCCL_NUM_PROTOCOLS];
  int buffSizes[NCCL_NUM_PROTOCOLS];
  void* mhandles[NCCL_NUM_PROTOCOLS];
  uint64_t step;
  uint64_t llLastCleaning;
  int netDeviceVersion;
  ncclNetDeviceType netDeviceType;
  ncclNetDeviceHandle_t* netDeviceHandle;
  
  // R2CC: Track accept log state
  int primaryAcceptStartLogged;
  int backupAcceptStartLogged;
  int connectInitLogged;

  // Variables related to the backup netDev
  void *netListenCommBackup;
  void *netRecvCommBackup;

  int netDevBackup;
  int useGdrBackup;
  int useDmaBufBackup;
  int needFlushBackup;
  int maxRecvsBackup;
  int netDeviceVersionBackup;
  ncclNetDeviceType netDeviceTypeBackup;
  ncclNetDeviceHandle_t *netDeviceHandleBackup;
  void *mhandlesBackup[NCCL_NUM_PROTOCOLS];

  int useBackup;
  int forceBackup;
  int forceBackupNotified;
  uint64_t lastFailoverEpoch;
  int waitFailoverReq;
  uint64_t waitFailoverStartMs;
  uint64_t waitFailoverLastWarnMs;
  uint64_t waitFailoverHintEpoch;
  uint64_t waitFailoverHintAbsStep;
  uint64_t waitFailoverHintLastSendMs;
  uint64_t waitFailoverHintSendCount;
  int state;
};

static bool r2ccTokenMatchesHca(const char* start, int len, const char* netName, int netDev) {
  if (len <= 0) return false;
  if (netName && (int)strlen(netName) == len && strncmp(start, netName, len) == 0) return true;
  // Numeric token: match netDev or suffix in netName (e.g. mlx5_2)
  int value = 0;
  bool numeric = true;
  for (int i = 0; i < len; i++) {
    if (!std::isdigit(static_cast<unsigned char>(start[i]))) { numeric = false; break; }
    value = value * 10 + (start[i] - '0');
  }
  if (!numeric) return false;
  if (value == netDev) return true;
  if (netName) {
    const char* underscore = strrchr(netName, '_');
    if (underscore && *(underscore + 1) != '\0') {
      int suffix = atoi(underscore + 1);
      if (suffix == value) return true;
    }
  }
  return false;
}

static bool r2ccMatchDisconnectedHca(int netDev, const char* netName) {
  const char* env = getenv("R2CC_DISCONNECTED_HCA");
  if (!env || env[0] == '\0') env = getenv("R2CC_Disconnected_HCA");
  if (!env || env[0] == '\0') return false;
  const char* p = env;
  while (*p) {
    while (*p == ' ' || *p == '\t' || *p == ',') p++;
    if (*p == '\0') break;
    const char* start = p;
    while (*p && *p != ' ' && *p != '\t' && *p != ',') p++;
    int len = (int)(p - start);
    if (r2ccTokenMatchesHca(start, len, netName, netDev)) return true;
  }
  return false;
}

/* Determine if two peers can communicate with NET */
static ncclResult_t canConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 1;
  if (info1->hostHash == info2->hostHash) {
    // If on the same host, check intra-node net is not disabled.
    NCCLCHECK(ncclTopoCheckNet(comm->topo, info1->rank, info2->rank, ret));
  }
  return ncclSuccess;
}

NCCL_PARAM(NetSharedBuffers, "NET_SHARED_BUFFERS", -2);
NCCL_PARAM(NetSharedComms, "NET_SHARED_COMMS", 1);

struct setupReq {
  int tpRank;
  int tpLocalRank;
  int tpRemoteRank;
  int shared;
  int netDev;
  int useGdr;
  int needFlush;
  int channelId;
  int connIndex;
};

NCCL_PARAM(RecvTimeout, "IB_TIMEOUT", 20);
NCCL_PARAM(RecvRetryCnt, "IB_RETRY_CNT", 7);
NCCL_PARAM(R2CCFailoverWaitMaxMs, "R2CC_FAILOVER_WAIT_MAX_MS", -1);


// Forward declaration
static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
static ncclResult_t sendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct setupReq req = { 0 };

  send->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  int proxyRank;
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &netId, &req.netDev, &proxyRank));
  INFO(NCCL_R2CC, "sendSetup: rank %d->%d, channel %d, got netDev=%d", 
       myInfo->rank, peerInfo->rank, channelId, req.netDev);
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, netId, 1, &req.useGdr));
  send->conn.flags |= req.useGdr ? NCCL_DIRECT_NIC : 0;

  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 1, proxyRank, &send->proxyConn));
  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  NCCLCHECK(ncclProxyCallBlocking(comm, &send->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), NULL, 0));

  if (proxyRank == myInfo->rank) {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  } else {
    INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s/%d(%d)%s%s", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name, req.netDev,
        proxyRank, req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  }
  *((int*)connectInfo) = comm->topParentRanks[proxyRank];
  return ncclSuccess;
}

// GDRCOPY support: TAIL_ENABLE When enabled locates the RX proxy tail in CUDA memory
NCCL_PARAM(GdrCopySyncEnable, "GDRCOPY_SYNC_ENABLE", 1);
// GDRCOPY support: FLUSH_ENABLE When enabled uses a PCI-E read to flush GDRDMA buffers
NCCL_PARAM(GdrCopyFlushEnable, "GDRCOPY_FLUSH_ENABLE", 0);

/* Setup recv connector */
static ncclResult_t recvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct setupReq req = { 0 };

  recv->conn.shared = req.shared = graph || connIndex == 0 ? 0 : ncclParamNetSharedBuffers() != -2 ? ncclParamNetSharedBuffers() : 1;
  req.channelId = channelId;
  req.connIndex = connIndex;

  // Use myInfo->rank as the receiver uses its own NIC
  int proxyRank;
  int64_t netId;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, myInfo->rank, &netId, &req.netDev, &proxyRank));
  INFO(NCCL_R2CC, "recvSetup: rank %d<-%d, channel %d, got netDev=%d from topology", 
       myInfo->rank, peerInfo->rank, channelId, req.netDev);
  
  // Debug: Also check what device sender would use
  int senderDev;
  int64_t senderNetId;
  int senderProxyRank;
  NCCLCHECK(ncclTopoGetNetDev(comm, myInfo->rank, graph, channelId, peerInfo->rank, &senderNetId, &senderDev, &senderProxyRank));
  INFO(NCCL_R2CC, "recvSetup: Checking sender's device selection - senderDev=%d, myDev=%d", senderDev, req.netDev);
  if (senderDev != req.netDev) {
    WARN("R2CC: Device mismatch! Receiver rank %d selected dev=%d, but sender rank %d would select dev=%d for channel %d", 
         myInfo->rank, req.netDev, peerInfo->rank, senderDev, channelId);
    INFO(NCCL_R2CC, "R2CC: Device mismatch detected - receiver uses dev=%d, sender would use dev=%d for channel %d", 
         req.netDev, senderDev, channelId);
  }
  
  NCCLCHECK(ncclTopoCheckGdr(comm->topo, myInfo->busId, netId, 0, &req.useGdr));

  // Determine whether we need to flush the GDR buffer on recv or not
  if (req.useGdr) NCCLCHECK(ncclTopoNeedFlush(comm->topo, myInfo->busId, &req.needFlush));

  // We don't support PXN on receive yet
  NCCLCHECK(ncclProxyConnect(comm, TRANSPORT_NET, 0, myInfo->rank, &recv->proxyConn));

  req.tpLocalRank = comm->topParentLocalRanks[comm->localRank];
  req.tpRank = comm->topParentRanks[myInfo->rank];
  req.tpRemoteRank = comm->topParentRanks[peerInfo->rank];
  TRACE(NCCL_INIT,"before recvSetup ncclProxyCallBlocking");
  NCCLCHECK(ncclProxyCallBlocking(comm, &recv->proxyConn, ncclProxyMsgSetup, &req, sizeof(req), connectInfo, 2*sizeof(ncclNetHandle_t)));
  TRACE(NCCL_INIT,"after recvSetup ncclProxyCallBlocking");
  INFO(NCCL_INIT|NCCL_NET,"Channel %02d/%d : %d[%d] -> %d[%d] [receive] via NET/%s/%d%s%s", channelId, connIndex, peerInfo->rank, peerInfo->nvmlDev, myInfo->rank, myInfo->nvmlDev, comm->ncclNet->name, req.netDev,
      req.useGdr ? "/GDRDMA" : "", req.shared ? "/Shared" : "");
  return ncclSuccess;
}

static ncclResult_t netMapShm(struct ncclComm *comm, struct connectMapMem* mem) {
  NCCLCHECK(ncclShmImportShareableBuffer(comm, &mem->createDesc, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr, &mem->attachDesc));
  return ncclSuccess;
}

static ncclResult_t netCreateShm(struct ncclProxyState* proxyState, struct connectMapMem* mem) {
  NCCLCHECK(ncclShmAllocateShareableBuffer(proxyState->tpRank, mem->size, false, &mem->createDesc, (void**)&mem->cpuPtr, (void**)&mem->gpuPtr));
  return ncclSuccess;
}

static ncclResult_t netDumpMap(struct connectMap* map) {
  printf("Dump map same process %d shared %d\n", map->sameProcess, map->shared);
  struct connectMapMem *mem = map->mems+NCCL_NET_MAP_HOSTMEM;
  printf("Mem 0: Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_DEVMEM;
  printf("Mem 1: Vid  mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_HOSTMEM;
  printf("Mem 2: Shared Host mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  mem = map->mems+NCCL_NET_MAP_SHARED_DEVMEM;
  printf("Mem 3: Shared Vid mem (%x B) CPU %p GPU %p\n", mem->size, mem->cpuPtr, mem->gpuPtr);
  printf("SendMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.sendMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, sendMem), map->offsets.sendMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem), NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem));
  printf("RecvMem -> Used %d Bank %d Offset %x, cpu %p gpu %p\n",
      map->offsets.recvMem & NCCL_NET_MAP_MASK_USED ? 1 : 0,
      NCCL_NET_MAP_OFFSET_BANK(map, recvMem), map->offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET,
      NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem), NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem));
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    printf("Proto %d -> Used %d Bank %d Offset %x, cpu %p, gpu %p\n", p,
        map->offsets.buffs[p] & NCCL_NET_MAP_MASK_USED ? 1 : 0,
        NCCL_NET_MAP_OFFSET_BANK(map, buffs[p]), map->offsets.buffs[p] & NCCL_NET_MAP_MASK_OFFSET,
        NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]), NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]));
  }
  printf("End of dump\n");
  return ncclSuccess;
}

struct netSendConnectArgs {
  ncclNetHandle_t handle;
};

struct netRecvConnectArgs {
  int proxyRank;
};

static ncclResult_t sendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  struct connectMap* map = (connectMap*) send->transportResources;

  void* opId;

  // map isn't allocated thus this op hasn't been submitted yet
  if (!map) {
    // Setup device pointers
    NCCLCHECK(ncclCalloc(&map, 1));
    send->transportResources = map;
    opId = send;
    INFO(NCCL_PROXY, "sendConnect ncclProxyCallAsync opId=%p", opId);
    netSendConnectArgs args[2] = {0};
    memcpy(&args, connectInfo, 2*sizeof(ncclNetHandle_t));
    TRACE(NCCL_INIT, "netSendConnectArgs size is %lu", sizeof(netSendConnectArgs));
    INFO(NCCL_R2CC, "sendConnect: Sending 2 handles to proxy, first at %p, second at %p", 
         &args[0], &args[1]);
    NCCLCHECK(ncclProxyCallAsync(comm, &send->proxyConn, ncclProxyMsgConnect, &args, 2*sizeof(netSendConnectArgs), sizeof(struct connectMap), opId));
  } else {
    opId =  send;
  }

  ncclResult_t ret;
  ret = ncclPollProxyResponse(comm, &send->proxyConn, map, opId);
  if (ret != ncclSuccess) {
    if (ret != ncclInProgress) {
      free(map);
      send->transportResources = NULL;
    }
    return ret;
  }
  INFO(NCCL_PROXY, "sendConnect ncclPollProxyResponse opId=%p", opId);

  if (map->sameProcess && !ncclCuMemEnable()) {
    if (map->cudaDev != comm->cudaDev) {
      // Enable P2P access for Legacy IPC
      cudaError_t err = cudaDeviceEnablePeerAccess(map->cudaDev, 0);
      if (err == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
      } else if (err != cudaSuccess) {
        WARN("failed to peer with device %d: %d %s", map->cudaDev, err, cudaGetErrorString(err));
        return ncclInternalError;
      }
    }
  } else if (!(map->sameProcess && map->cudaDev == comm->cudaDev)) {
    if (!map->sameProcess) NCCLCHECK(netMapShm(comm, map->mems + NCCL_NET_MAP_HOSTMEM));
    if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
      map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr = NULL;
      NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.rank,
                                             map->mems[NCCL_NET_MAP_DEVMEM].size,
                                             &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                             (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = NULL;
    }
    if (map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size) {
      void** sharedDevMemPtr = comm->proxyState->sharedDevMems + send->proxyConn.tpLocalRank;
      if (*sharedDevMemPtr == NULL) {
        map->mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = NULL;
        NCCLCHECK(ncclP2pImportShareableBuffer(comm, send->proxyConn.rank,
                                               map->mems[NCCL_NET_MAP_SHARED_DEVMEM].size,
                                               &map->mems[NCCL_NET_MAP_SHARED_DEVMEM].ipcDesc,
                                               sharedDevMemPtr));
      }
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].gpuPtr = (char*)(*sharedDevMemPtr);
      map->mems[NCCL_NET_MAP_SHARED_DEVMEM].cpuPtr = NULL;
    }
  }
  //NCCLCHECK(netDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  send->conn.head = gdcMem ? (uint64_t*)gdcMem : &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  send->conn.tail = &recvMem->tail;
  send->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  send->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  for (int i=0; i<NCCL_STEPS; i++) {
    send->conn.connFifo[i].offset = -1;
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    send->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  if (send->proxyConn.sameProcess) {
    if (send->proxyConn.connection->netDeviceHandle) {
      send->conn.netDeviceHandle = *send->proxyConn.connection->netDeviceHandle;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        send->conn.mhandles[p] = send->proxyConn.connection->mhandles[p];
    }

    if (send->proxyConn.connection->needsProxyProgress) {
      send->proxyConn.proxyProgress = sendProxyProgress;
    } else {
      send->proxyConn.proxyProgress = NULL;
    }
  } else {
    send->proxyConn.proxyProgress = sendProxyProgress;
  }

  return ncclSuccess;
}

// Forward declare
static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args);

/* Connect to this peer */
static ncclResult_t recvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  struct connectMap* map = (connectMap*) recv->transportResources;
  void* opId;
  if (!map) {
    NCCLCHECK(ncclCalloc(&map, 1));
    recv->transportResources = map;
    // Use recv connector as unique identifier
    opId = recv;
    INFO(NCCL_PROXY, "recvConnect ncclProxyCallAsync opId=%p &recv->proxyConn=%p connectInfo=%p",
       opId, &recv->proxyConn, connectInfo);
    netRecvConnectArgs args = {0};
    args.proxyRank = *((int*)connectInfo);
    NCCLCHECK(ncclProxyCallAsync(comm, &recv->proxyConn, ncclProxyMsgConnect, &args, sizeof(netRecvConnectArgs), sizeof(struct connectMap), opId));
  } else {
    opId = recv;
  }

  ncclResult_t ret;
  NCCLCHECK(ret = ncclPollProxyResponse(comm, &recv->proxyConn, map, opId));
  if (ret != ncclSuccess) {
    if (ret != ncclInProgress) {
      free(map);
      recv->transportResources = NULL;
    }
    return ret;
  }
  INFO(NCCL_PROXY, "recvConnect ncclPollProxyResponse opId=%p", opId);
  //NCCLCHECK(netDumpMap(map));

  struct ncclSendMem *sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, sendMem);
  recv->conn.head = &sendMem->head;

  struct ncclRecvMem *recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, gpu, recvMem);
  void* gdcMem = map->mems[NCCL_NET_MAP_GDCMEM].gpuPtr;
  recv->conn.tail = gdcMem ? (uint64_t*)gdcMem : &recvMem->tail;
  recv->conn.stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;
  recv->conn.connFifo = recvMem->connFifo;
  // Only fuse P2P buffers, continue to allocate dedicated buffers for ring/tree
  for (int i=0; i<NCCL_STEPS; i++) {
    recvMem->connFifo[i].mode = map->shared ? NCCL_MODE_OFFSET : NCCL_MODE_NORMAL;
  }

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
    recv->conn.buffs[p] = NCCL_NET_MAP_GET_POINTER(map, gpu, buffs[p]);

  if (recv->proxyConn.sameProcess) {
    if (recv->proxyConn.connection->netDeviceHandle) {
      recv->conn.netDeviceHandle = *recv->proxyConn.connection->netDeviceHandle;

      for (int p=0; p<NCCL_NUM_PROTOCOLS; p++)
        recv->conn.mhandles[p] = recv->proxyConn.connection->mhandles[p];
    }

    if (recv->proxyConn.connection->needsProxyProgress) {
      recv->proxyConn.proxyProgress = recvProxyProgress;
    } else {
      recv->proxyConn.proxyProgress = NULL;
    }
  } else {
    recv->proxyConn.proxyProgress = recvProxyProgress;
  }

  return ncclSuccess;
}

static ncclResult_t sendFree(struct ncclConnector* send) {
  struct connectMap* map = (struct connectMap*)(send->transportResources);
  if (map) {
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
    if (map->cudaDev != cudaDev && map->mems[NCCL_NET_MAP_DEVMEM].size) {
      if (ncclCuMemEnable()) {
        // cuMem API support
        NCCLCHECK(ncclP2pFreeShareableBuffer(&map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
        NCCLCHECK(ncclCuMemFree(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        // Legacy CUDA IPC support
        CUDACHECK(cudaIpcCloseMemHandle(map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      }
    }
    if (!map->sameProcess) {
      NCCLCHECK(ncclShmIpcClose(&map->mems[NCCL_NET_MAP_HOSTMEM].attachDesc));
    }
    free(map);
  }

  return ncclSuccess;
}

static ncclResult_t recvFree(struct ncclConnector* recv) {
  if (recv->transportResources) free(recv->transportResources);
  return ncclSuccess;
}

#define NCCL_SHARED_STEPS 16
static ncclResult_t sharedNetBuffersInit(struct ncclProxyState* proxyState, int cuda, int tpLocalRank, int type, int sameProcess,
    int nChannels, char** gpuPtr, char** cpuPtr, int* size, ncclIpcDesc *ipcDesc) {
  if (cuda == 0 && sameProcess == 0) {
      WARN("PXN should not use host buffers for data");
      return ncclInternalError;
  }
  struct ncclProxyProgressState* progressState = &proxyState->progressState;
  if (progressState->localPeers == NULL) {
    NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
  }
  struct ncclProxyPeer** localPeers = progressState->localPeers;
  if (localPeers[tpLocalRank] == NULL) {
    NCCLCHECK(ncclCalloc(localPeers + tpLocalRank, 1));
  }
  struct ncclProxyPeer* peer = localPeers[tpLocalRank];
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  state->refcount++;
  if (state->size == 0) {
    state->size = nChannels * NCCL_SHARED_STEPS * proxyState->p2pChunkSize;
  }

  if (size) *size = state->size;

  if (cuda && state->cudaBuff == NULL) {
    if (sameProcess == 0 || ncclCuMemEnable()) {
      NCCLCHECK(ncclP2pAllocateShareableBuffer(state->size, 0, &state->ipcDesc, (void**)&state->cudaBuff));
    } else {
      NCCLCHECK(ncclCudaCalloc(&state->cudaBuff, state->size));
    }
  }
  if (!cuda && state->hostBuff == NULL) {
    NCCLCHECK(ncclCudaHostCalloc(&state->hostBuff, state->size));
  }
  if (cpuPtr) *cpuPtr = cuda ? state->cudaBuff : state->hostBuff;
  if (gpuPtr) *gpuPtr = (cpuPtr && sameProcess) ? *cpuPtr : NULL;
  if (ipcDesc) memcpy(ipcDesc, &state->ipcDesc, sizeof(state->ipcDesc));
  return ncclSuccess;
}

static ncclResult_t sharedBuffersGet(struct ncclProxyState* proxyState, int channel, int slot, int* offset, int* size) {
  // Use different pools for different channels and also separate send/recv.
  int globalSlot = (channel*NCCL_SHARED_STEPS)+slot;
  *offset = proxyState->p2pChunkSize * globalSlot;
  if (size) *size = proxyState->p2pChunkSize;
  return ncclSuccess;
}

static ncclResult_t sharedNetBuffersDestroy(struct ncclProxyState* proxyState, int tpLocalRank, int type, struct ncclProxyConnection* connection) {
  if (proxyState->progressState.localPeers == NULL) NCCLCHECK(ncclInternalError);
  struct ncclProxyPeer* peer = proxyState->progressState.localPeers[tpLocalRank];
  if (peer == NULL) NCCLCHECK(ncclInternalError);
  struct ncclProxySharedP2p* state = type == 0 ? &peer->send : &peer->recv;
  if (state->size == 0) NCCLCHECK(ncclInternalError);
  if (ncclAtomicRefCountDecrement(&state->refcount) == 0) {
    if (state->cudaBuff) {
      if (!connection->sameProcess || ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&state->ipcDesc));
      }
      NCCLCHECK(ncclCudaFree(state->cudaBuff));
    }
    if (state->hostBuff) NCCLCHECK(ncclCudaHostFree(state->hostBuff));
  }

  if (peer->send.refcount || peer->recv.refcount) return ncclSuccess;

  free(peer);
  proxyState->progressState.localPeers[tpLocalRank] = NULL;
  for (int r = 0; r < proxyState->tpLocalnRanks; r++) {
    if (proxyState->progressState.localPeers[r]) return ncclSuccess;
  }
  // All peers are freed, free array
  free(proxyState->progressState.localPeers);
  proxyState->progressState.localPeers = NULL;
  return ncclSuccess;
}

static ncclResult_t proxySharedInit(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, int nChannels) {
  NCCLCHECK(sharedNetBuffersInit(proxyState, 1, connection->tpLocalRank, 0, connection->sameProcess, nChannels, NULL, NULL, NULL, NULL));
  return ncclSuccess;
}

static ncclResult_t sendProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  // R2CC: Track setup calls
  static int sendSetupCount = 0;
  sendSetupCount++;
  
  // Check for extra setup calls
  static int sendExtraDetected = 0;
  if (sendSetupCount > 24 && !sendExtraDetected) {
    sendExtraDetected = 1;
    INFO(NCCL_R2CC, "WARNING: Extra sendProxySetup detected! count=%d, channel=%d, netDev=%d", 
         sendSetupCount, req->channelId, req->netDev);
  }
  
  struct sendNetResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  // Initialize common variables
  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->shared = connection->shared = req->shared;
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  
  // R2CC: Initialize log state tracking
  resources->primaryConnStartLogged = 0;
  resources->backupConnStartLogged = 0;
  resources->connectInitLogged = 0;

  // Initialize default netDev variable
  resources->netDev = req->netDev;
  resources->useGdr = req->useGdr;
  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxRecvs = props.maxRecvs;
  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;

  // Initialize backup netDev variables (assuming req->netDev^1, same req->useGdr, etc.)
  int nNetDevs = 0;
  NCCLCHECK(proxyState->ncclNet->devices(&nNetDevs));
  // if (nNetDevs > 1) {
  //   resources->netDevBackup = (req->netDev == 0) ? 1 : 0;
  // } else {
  //   resources->netDevBackup = req->netDev;
  // }
  if (nNetDevs % 2 == 0)
    resources->netDevBackup = req->netDev^1;
  else{
    if(req->netDev == nNetDevs - 1){
      resources->netDevBackup = req->netDev; // demo doesn't support odd number of devices
    } else {
      resources->netDevBackup = req->netDev^1; 
    }
  }

  resources->useGdrBackup = req->useGdr;
  
  INFO(NCCL_R2CC, "sendProxySetup: Channel %d, netDev=%d, netDevBackup=%d", 
       resources->channelId, resources->netDev, resources->netDevBackup);
  
  ncclNetProperties_t propsBackup;
  NCCLCHECK(proxyState->ncclNet->getProperties(resources->netDevBackup, &propsBackup));
  resources->useDmaBufBackup = resources->useGdrBackup && proxyState->dmaBufSupport && (propsBackup.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxRecvsBackup = propsBackup.maxRecvs;
  resources->netDeviceVersionBackup = propsBackup.netDeviceVersion;
  resources->netDeviceTypeBackup = propsBackup.netDeviceType;

  const char* r2ccMode = getenv("R2CC_MODE");
  if (r2ccMode && atoi(r2ccMode) == 1) {
    // Simulate disable of device 1 (second device)
    if (resources->netDev == 0) {
      resources->useBackup = 1;
      // Get device names and log
      INFO(NCCL_R2CC, "R2CC_MODE=1 (SEND): Channel %d will simulate disable of device %d (%s) and use backup device %d (%s)", 
           resources->channelId, resources->netDev, props.name, resources->netDevBackup, propsBackup.name);
    }
  }

  resources->forceBackup = 0;
  resources->forceBackupNotified = 0;
  resources->stepSyncRequested = 0;
  resources->stepSyncWaitIters = 0;
  resources->failoverEpoch = 0;
  resources->failoverWaitAck = 0;
  resources->failoverReqAbsStep = 0;
  resources->failoverWaitStartMs = 0;
  resources->failoverWaitLastWarnMs = 0;
  resources->stallReason = 0;
  resources->stallStartMs = 0;
  resources->stallLastWarnMs = 0;
  resources->stallPosted = 0;
  resources->stallTransmitted = 0;
  resources->stallDone = 0;
  if (r2ccMatchDisconnectedHca(resources->netDev, props.name)) {
    resources->forceBackup = 1;
    INFO(NCCL_R2CC, "R2CC_DISCONNECTED_HCA: RECV channel %d primary dev=%d (%s) will inject failure at step 10",
         resources->channelId, resources->netDev, props.name);
  }

  resources->forceBackup = 0;
  resources->forceBackupNotified = 0;
  if (r2ccMatchDisconnectedHca(resources->netDev, props.name)) {
    resources->forceBackup = 1;
    INFO(NCCL_R2CC, "R2CC_DISCONNECTED_HCA: SEND channel %d primary dev=%d (%s) will inject failure at step 10",
         resources->channelId, resources->netDev, props.name);
  }

  // We don't return any data
  if (respSize != 0) return ncclInternalError;
  *done = 1;
  return ncclSuccess;
}

static ncclResult_t recvProxySetup(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct setupReq* req = (struct setupReq*) reqBuff;
  if (reqSize != sizeof(struct setupReq)) return ncclInternalError;

  struct recvNetResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  connection->transportResources = resources;

  resources->tpRank = req->tpRank;
  resources->tpLocalRank = req->tpLocalRank;
  resources->tpRemoteRank = req->tpRemoteRank;
  resources->netDev = req->netDev;
  resources->shared = connection->shared = req->shared;
  resources->useGdr = req->useGdr;
  resources->needFlush = req->needFlush;
  resources->channelId = req->channelId;
  resources->connIndex = req->connIndex;
  
  // R2CC: Initialize log state tracking
  resources->primaryAcceptStartLogged = 0;
  resources->backupAcceptStartLogged = 0;

  ncclNetProperties_t props;
  NCCLCHECK(proxyState->ncclNet->getProperties(req->netDev, &props));
  /* DMA-BUF support */
  resources->useDmaBuf = resources->useGdr && proxyState->dmaBufSupport && (props.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxRecvs = props.maxRecvs;
  resources->netDeviceVersion = props.netDeviceVersion;
  resources->netDeviceType = props.netDeviceType;


  // Initialize backup netDev variables
  int nNetDevs = 0;
  NCCLCHECK(proxyState->ncclNet->devices(&nNetDevs));
  // if (nNetDevs > 1) {
  //   resources->netDevBackup = (req->netDev == 0) ? 1 : 0;
  // } else {
  //   resources->netDevBackup = req->netDev;
  // }

  if (nNetDevs % 2 == 0)
    resources->netDevBackup = req->netDev^1;
  else{
    if(req->netDev == nNetDevs - 1){
      resources->netDevBackup = req->netDev; // demo doesn't support odd number of devices
    } else {
      resources->netDevBackup = req->netDev^1; 
    }
  }

  resources->useGdrBackup = req->useGdr;
  resources->needFlushBackup = req->needFlush;
  
  INFO(NCCL_R2CC, "recvProxySetup: Channel %d, netDev=%d, netDevBackup=%d", 
       resources->channelId, resources->netDev, resources->netDevBackup);
  
    // Get properties for backup netDev
  ncclNetProperties_t propsBackup;
  NCCLCHECK(proxyState->ncclNet->getProperties(resources->netDevBackup, &propsBackup));
  resources->useDmaBufBackup = resources->useGdrBackup && proxyState->dmaBufSupport && (propsBackup.ptrSupport & NCCL_PTR_DMABUF);
  resources->maxRecvsBackup = propsBackup.maxRecvs;
  resources->netDeviceVersionBackup = propsBackup.netDeviceVersion;
  resources->netDeviceTypeBackup = propsBackup.netDeviceType;

  const char* r2ccMode = getenv("R2CC_MODE");
  if (r2ccMode && atoi(r2ccMode) == 1) {
    // Simulate disable of device 1 (second device)
  if (resources->netDev == 0) {
      resources->useBackup = 1;
      // Get device names and log
      INFO(NCCL_R2CC, "R2CC_MODE=1 (RECV): Channel %d will simulate disable of device %d (%s) and use backup device %d (%s)", 
           resources->channelId, resources->netDev, props.name, resources->netDevBackup, propsBackup.name);
    }
  }

  resources->lastFailoverEpoch = 0;
  resources->waitFailoverReq = 0;
  resources->waitFailoverStartMs = 0;
  resources->waitFailoverLastWarnMs = 0;
  resources->waitFailoverHintEpoch = 0;
  resources->waitFailoverHintAbsStep = 0;
  resources->waitFailoverHintLastSendMs = 0;
  resources->waitFailoverHintSendCount = 0;

  TRACE(NCCL_INIT, "listen 1");
  // if (respSize != sizeof(ncclNetHandle_t)) return ncclInternalError;
  
  // R2CC: Log detailed listen setup
  static int setupCount = 0;
  setupCount++;
  
  // R2CC: Check if this is being called after communicator creation
  static int lastCommCount = 0;
  static int extraSetupDetected = 0;
  int currentCommCount = 0; // This would ideally come from comm object if available
  
  // Simple heuristic: if setupCount > 12*2 (6 channels * 2 comms * 2 for send/recv), it's extra
  if (setupCount > 24 && !extraSetupDetected) {
    extraSetupDetected = 1;
    INFO(NCCL_R2CC, "WARNING: Extra recvProxySetup detected after communicator creation! setupCount=%d", setupCount);
  }
  
  INFO(NCCL_R2CC, "recvProxySetup[%d]: START channel=%d, primary dev=%d, backup dev=%d%s", 
       setupCount, resources->channelId, resources->netDev, resources->netDevBackup,
       extraSetupDetected ? " [EXTRA]" : "");
  
  // R2CC DEBUG: Log before first listen
  INFO(NCCL_R2CC, "DEBUG: Channel %d calling listen for PRIMARY dev=%d", 
       resources->channelId, resources->netDev);
  NCCLCHECK(proxyState->ncclNet->listen(resources->netDev, respBuff, &resources->netListenComm));
  INFO(NCCL_R2CC, "DEBUG: Channel %d PRIMARY listen SUCCESS, listenComm=%p", 
       resources->channelId, resources->netListenComm);
  
  // R2CC: Log primary handle details with Connection prefix
  ncclNetHandle_t* primaryHandle = (ncclNetHandle_t*)respBuff;
  uint8_t* h1bytes = (uint8_t*)primaryHandle;
  INFO(NCCL_R2CC, "Connection: ListenComm at rank=%d channel=%d type=PRIMARY dev=%d listenComm=%p handle=[%02x%02x%02x%02x%02x%02x%02x%02x]",
       resources->tpRank, resources->channelId, resources->netDev, resources->netListenComm,
       h1bytes[0], h1bytes[1], h1bytes[2], h1bytes[3], h1bytes[4], h1bytes[5], h1bytes[6], h1bytes[7]);
  
  TRACE(NCCL_INIT, "listen 2");

  // R2CC: Create backup listen socket (now protected by mutex in ncclIbListen)
  INFO(NCCL_R2CC, "DEBUG: Channel %d calling listen for BACKUP dev=%d", 
       resources->channelId, resources->netDevBackup);
  NCCLCHECK(proxyState->ncclNet->listen(resources->netDevBackup,
                                         ((char*)respBuff) + sizeof(ncclNetHandle_t),
                                         &resources->netListenCommBackup));
  INFO(NCCL_R2CC, "DEBUG: Channel %d BACKUP listen SUCCESS, listenCommBackup=%p", 
       resources->channelId, resources->netListenCommBackup);
  
  // R2CC: Log backup handle details with Connection prefix
  ncclNetHandle_t* backupHandle = (ncclNetHandle_t*)(((char*)respBuff) + sizeof(ncclNetHandle_t));
  uint8_t* h2bytes = (uint8_t*)backupHandle;
  INFO(NCCL_R2CC, "Connection: ListenComm at rank=%d channel=%d type=BACKUP dev=%d listenComm=%p handle=[%02x%02x%02x%02x%02x%02x%02x%02x]",
       resources->tpRank, resources->channelId, resources->netDevBackup, resources->netListenCommBackup,
       h2bytes[0], h2bytes[1], h2bytes[2], h2bytes[3], h2bytes[4], h2bytes[5], h2bytes[6], h2bytes[7]);
  
  // Debug: Print handle addresses and content
  ncclNetHandle_t* handle1 = (ncclNetHandle_t*)respBuff;
  ncclNetHandle_t* handle2 = (ncclNetHandle_t*)(((char*)respBuff) + sizeof(ncclNetHandle_t));
  INFO(NCCL_R2CC, "recvProxySetup[%d]: Channel %d created handles - primary at %p, backup at %p", 
       setupCount, resources->channelId, handle1, handle2);
  
  // Debug: Log first 8 bytes of each handle as identifier
  uint64_t* h1 = (uint64_t*)handle1;
  uint64_t* h2 = (uint64_t*)handle2;
  INFO(NCCL_R2CC, "recvProxySetup[%d]: Channel %d handle IDs - primary=%lx, backup=%lx", 
       setupCount, resources->channelId, *h1, *h2);

  // char line[SOCKET_NAME_MAXLEN+1];
  // char line2[SOCKET_NAME_MAXLEN+1];
  // if(resources->channelId == 0){
  //   TRACE(NCCL_INIT, "resources->channelId == 0 && check the handle 1 addr %s magic %lu", ncclSocketToString((ncclSocketAddress*)respBuff, line), *((uint64_t*)((char*)respBuff + sizeof(ncclSocketAddress))));
  //   TRACE(NCCL_INIT, "resources->channelId == 0 && check the handle 2 addr %s magic %lu", ncclSocketToString((ncclSocketAddress*)((char*)respBuff+128), line2), *((uint64_t*)((char*)respBuff+128) +sizeof(ncclSocketAddress) ));
  // }
  // TRACE(NCCL_INIT, "sizeof ncclNetHandle_t %lu, sizeof ncclConnect %lu", sizeof(ncclNetHandle_t), sizeof(ncclConnect));

  *done = 1;

  return ncclSuccess;
}

// This function embeds plugin-specific rules given the current versions
static ncclResult_t ncclNetGetDeviceHandle(ncclNetDeviceType type, int version, bool isRecv, ncclNetDeviceHandle_t** handle) {
  bool needsDeviceHandle  = false;

  if (type == NCCL_NET_DEVICE_UNPACK) {
    if (version == NCCL_NET_DEVICE_UNPACK_VERSION && isRecv) {
      needsDeviceHandle  = true;
    }
  }

  // Don't re-alloc netDeviceHandles
  if (needsDeviceHandle && (*handle == NULL)) {
    NCCLCHECK(ncclCalloc(handle, 1));
    (*handle)->netDeviceType = type;
    (*handle)->netDeviceVersion = version;
  } else if (!needsDeviceHandle) {
    *handle = NULL;
  }

  return ncclSuccess;
}

static ncclResult_t sendProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  resources->useBackup = 0;
  
  // R2CC: Log detailed connect info
  static int connectCount = 0;
  connectCount++;
  
  // Check R2CC_MODE environment variable
  const char* r2ccMode = getenv("R2CC_MODE");
  if (r2ccMode && atoi(r2ccMode) == 1) {
    // Simulate disable of device 1 (second device)
    if (resources->netDev == 0) {
      resources->useBackup = 1;
      INFO(NCCL_R2CC, "R2CC_MODE=1 (SEND-CONNECT): Channel %d will use backup for device %d", 
           resources->channelId, resources->netDev);
    }
  }
  // if (reqSize != sizeof(netSendConnectArgs)) return ncclInternalError;
  ncclResult_t ret = ncclSuccess;
  ncclResult_t ret2 = ncclSuccess;
  netSendConnectArgs* req = (netSendConnectArgs*) reqBuff;
  
  // R2CC: Log received handles only once at the beginning
  if (!resources->connectInitLogged) {
    uint8_t* hbytes = (uint8_t*)req->handle;
    uint8_t* hbytes2 = (uint8_t*)((req+1)->handle);
    INFO(NCCL_R2CC, "Connection: Connecting from rank=%d channel=%d type=PRIMARY dev=%d to handle=[%02x%02x%02x%02x%02x%02x%02x%02x] remoteRank=%d",
         resources->tpRank, resources->channelId, resources->netDev,
         hbytes[0], hbytes[1], hbytes[2], hbytes[3], hbytes[4], hbytes[5], hbytes[6], hbytes[7],
         resources->tpRemoteRank);
    INFO(NCCL_R2CC, "Connection: Connecting from rank=%d channel=%d type=BACKUP dev=%d to handle=[%02x%02x%02x%02x%02x%02x%02x%02x] remoteRank=%d",
         resources->tpRank, resources->channelId, resources->netDevBackup,
         hbytes2[0], hbytes2[1], hbytes2[2], hbytes2[3], hbytes2[4], hbytes2[5], hbytes2[6], hbytes2[7],
         resources->tpRemoteRank);
    resources->connectInitLogged = 1;
  }
  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceType, resources->netDeviceVersion, false /*isRecv*/, &resources->netDeviceHandle));
  if (resources->shared) {
    // Shared buffers
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->tpLocalRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers + resources->tpLocalRank, 1));
    }
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->send.proxyAppend + resources->channelId;

    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      if (progressState->netComms[resources->netDev] == NULL) {
        NCCLCHECK(ncclCalloc(progressState->netComms + resources->netDev, proxyState->tpnRanks));
      }
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev] + resources->tpRemoteRank;
      if (comms->sendComm[resources->channelId] == NULL) ret = proxyState->ncclNet->connect(resources->netDev, req->handle, comms->sendComm + resources->channelId, &resources->netDeviceHandle);
      resources->netSendComm = comms->sendComm[resources->channelId];
      if (comms->sendComm[resources->channelId]) comms->sendRefCount[resources->channelId]++;
    } else {
      ret = proxyState->ncclNet->connect(resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    }
  } else {
    // Connect to remote peer
    // TRACE(NCCL_INIT, "before netSendComm");
    // ret = proxyState->ncclNet->connect(resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    // TRACE(NCCL_INIT, "after netSendComm, before netSendCommBackup");
    // TRACE(NCCL_INIT, "ret = %d", ret);

    // ret2 = proxyState->ncclNet->connect(resources->netDevBackup, (req+1)->handle, &resources->netSendCommBackup, &resources->netDeviceHandleBackup);
    // TRACE(NCCL_INIT, "after netSendCommBackup");
    // TRACE(NCCL_INIT, "ret2 = %d", ret2);
    connection->proxyAppendPtr = &connection->proxyAppend;
  }
  // char line[SOCKET_NAME_MAXLEN+1];
  // char line2[SOCKET_NAME_MAXLEN+1];
  // if(resources->channelId == 0){
  //   TRACE(NCCL_INIT, "resources->channelId == 0 && check the handle 1 addr %s magic %lu", ncclSocketToString((ncclSocketAddress*)req, line), *((uint64_t*)((char*)req + sizeof(ncclSocketAddress))));
  //   TRACE(NCCL_INIT, "resources->channelId == 0 && check the handle 2 addr %s magic %lu", ncclSocketToString((ncclSocketAddress*)(req+1), line2), *((uint64_t*)((char*)(req+1) +sizeof(ncclSocketAddress) )));
  // }



  // NCCLCHECK(ret);
  // NCCLCHECK(ret2);

  // if(resources->netSendComm == NULL && resources->netSendCommBackup == NULL){
  //   TRACE(NCCL_INIT, "default dev %d == NULL, backup  dev %d == NULL channelId %d", resources->netDev, resources->netDevBackup, resources->channelId);
  // }

  // if(resources->netSendComm != NULL && resources->netSendCommBackup == NULL){
  //   TRACE(NCCL_INIT, "default dev %d done, backup  dev %d == NULL channelId %d", resources->netDev, resources->netDevBackup, resources->channelId);
  // }
  //   if(resources->netSendComm == NULL && resources->netSendCommBackup != NULL){
  //   TRACE(NCCL_INIT, "default dev %d == NULL, backup  dev %d done channelId %d", resources->netDev, resources->netDevBackup, resources->channelId);
  // }

  // if (resources->netSendComm == NULL) {
  // //if (resources->netSendComm == NULL || resources->netSendCommBackup == NULL) {
  //   *done = 0;
  //   return ncclInProgress;
  // }

  // R2CC: Parallel connect - try both connections simultaneously
  int primaryDone = 0;
  int backupDone = 0;
  
  // Try to connect PRIMARY
  if (resources->netSendComm == NULL) {
    if (!resources->primaryConnStartLogged) {
      uint8_t* hbytes = (uint8_t*)req->handle;
      INFO(NCCL_R2CC, "Connection: Connect START rank=%d channel=%d type=PRIMARY dev=%d to handle=[%02x%02x%02x%02x%02x%02x%02x%02x]", 
           resources->tpRank, resources->channelId, resources->netDev,
           hbytes[0], hbytes[1], hbytes[2], hbytes[3], hbytes[4], hbytes[5], hbytes[6], hbytes[7]);
      resources->primaryConnStartLogged = 1;
    }
    // R2CC DEBUG: Log before connect
    INFO(NCCL_R2CC, "DEBUG: Channel %d calling connect for PRIMARY dev=%d, handle=%p", 
         resources->channelId, resources->netDev, req->handle);
    ret = proxyState->ncclNet->connect(resources->netDev, req->handle, &resources->netSendComm, &resources->netDeviceHandle);
    INFO(NCCL_R2CC, "DEBUG: Channel %d PRIMARY connect returned %d, sendComm=%p", 
         resources->channelId, ret, resources->netSendComm);
    NCCLCHECK(ret);
    if (resources->netSendComm != NULL) {
      INFO(NCCL_R2CC, "Connection: Connect COMPLETED rank=%d channel=%d type=PRIMARY dev=%d sendComm=%p", 
           resources->tpRank, resources->channelId, resources->netDev, resources->netSendComm);
      primaryDone = 1;
    }
  } else {
    primaryDone = 1;
  }

  // Try to connect BACKUP (parallel with primary)
  if (resources->netSendCommBackup == NULL) {
    if (!resources->backupConnStartLogged) {
      uint8_t* hbytes2 = (uint8_t*)((req+1)->handle);
      INFO(NCCL_R2CC, "Connection: Connect START rank=%d channel=%d type=BACKUP dev=%d to handle=[%02x%02x%02x%02x%02x%02x%02x%02x]", 
           resources->tpRank, resources->channelId, resources->netDevBackup,
           hbytes2[0], hbytes2[1], hbytes2[2], hbytes2[3], hbytes2[4], hbytes2[5], hbytes2[6], hbytes2[7]);
      resources->backupConnStartLogged = 1;
    }
    // R2CC DEBUG: Log before backup connect
    INFO(NCCL_R2CC, "DEBUG: Channel %d calling connect for BACKUP dev=%d, handle=%p", 
         resources->channelId, resources->netDevBackup, (req+1)->handle);
    ret2 = proxyState->ncclNet->connect(resources->netDevBackup, (req+1)->handle, &resources->netSendCommBackup, &resources->netDeviceHandleBackup);
    INFO(NCCL_R2CC, "DEBUG: Channel %d BACKUP connect returned %d, sendCommBackup=%p", 
         resources->channelId, ret2, resources->netSendCommBackup);
    NCCLCHECK(ret2);
    if (resources->netSendCommBackup != NULL) {
      INFO(NCCL_R2CC, "Connection: Connect COMPLETED rank=%d channel=%d type=BACKUP dev=%d sendComm=%p", 
           resources->tpRank, resources->channelId, resources->netDevBackup, resources->netSendCommBackup);
      backupDone = 1;
    }
  } else {
    backupDone = 1;
  }

  // Check if both connections are complete
  if (primaryDone && backupDone) {
    *done = 1;
    INFO(NCCL_R2CC, "PARALLEL CONNECT: Channel %d - Both connections COMPLETE", resources->channelId);
  } else {
    *done = 0;
    const char* primaryStatus = primaryDone ? "DONE" : "CONNECTING";
    const char* backupStatus = backupDone ? "DONE" : "CONNECTING";
    INFO(NCCL_R2CC, "PARALLEL CONNECT: Channel %d - PRIMARY=%s, BACKUP=%s", 
         resources->channelId, primaryStatus, backupStatus);
    return ncclInProgress;
  }
  TRACE(NCCL_INIT, "sendProxyConnect done with two comm channelId %d", resources->channelId);

  if (resources->netDeviceHandle) {
    connection->netDeviceHandle = resources->netDeviceHandle;
    connection->needsProxyProgress = connection->netDeviceHandle->needsProxyProgress;
  } else {
    connection->needsProxyProgress = 1;
  }

  if (resources->netDeviceHandleBackup) {
    connection->netDeviceHandleBackup = resources->netDeviceHandleBackup;
    connection->needsProxyProgress = connection->netDeviceHandleBackup->needsProxyProgress;
  } else {
    connection->needsProxyProgress = 1;
  }



  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess = connection->sameProcess;
  map->shared = resources->shared;
  CUDACHECK(cudaGetDevice(&map->cudaDev));

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, p!= NCCL_PROTO_LL && resources->useGdr, proxyState->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(sharedNetBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 0, map->sameProcess, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, &mapMem->ipcDesc));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;

    if (proxyState->allocP2pNetLLBuffers) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, 0 /*p == NCCL_PROTO_LL*/, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
      resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
    }

    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (!map->sameProcess || ncclCuMemEnable()) {
        ALIGN_SIZE(map->mems[NCCL_NET_MAP_DEVMEM].size, CUDA_IPC_MIN);
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size));
      }
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  if (map->sameProcess) {
    NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
    map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  } else {
    NCCLCHECK(netCreateShm(proxyState, map->mems+NCCL_NET_MAP_HOSTMEM));
    void* sendMem = (void*)NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
    void* recvMem = (void*)NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
    memset(sendMem, 0, sizeof(struct ncclSendMem));
    memset(recvMem, 0, sizeof(struct ncclRecvMem));
  }
  if (ncclGdrCopy && map->sameProcess && ncclParamGdrCopySyncEnable()) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 1, &resources->gdrDesc));

    resources->gdcSync = cpuPtr;
    struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
    gdcMem->cpuPtr = (char*)cpuPtr;
    gdcMem->gpuPtr = (char*)gpuPtr;
    gdcMem->size = sizeof(uint64_t); // sendMem->head
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);

  // Don't give credits yet in shared mode.
  (resources->gdcSync ? *resources->gdcSync : resources->sendMem->head) =
    (map->shared ? -NCCL_STEPS : 0);
  for (int i=0; i<NCCL_STEPS; i++) resources->recvMem->connFifo[i].size = -1;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        int dmabuf_fd;
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netSendCommBackup, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandlesBackup[p]));
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
#endif
      {
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netSendComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netSendCommBackup, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandlesBackup[p]));
      }

      // Copy the mhandle dptr, if implemented
      if (resources->netDeviceHandle && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netSendComm, resources->mhandles[p], &connection->mhandles[p]));
      if (resources->netDeviceHandleBackup && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netSendCommBackup, resources->mhandlesBackup[p], &connection->mhandlesBackup[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  memcpy(respBuff, map, sizeof(struct connectMap));
  return ncclSuccess;
}

static ncclResult_t recvProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  if (reqSize != sizeof(netRecvConnectArgs)) return ncclInternalError;
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  netRecvConnectArgs* req = (netRecvConnectArgs*) reqBuff;
  resources->tpRemoteProxyRank = req->proxyRank;
  resources->useBackup = 0;
  
  // Check R2CC_MODE environment variable
  const char* r2ccMode = getenv("R2CC_MODE");
  if (r2ccMode && atoi(r2ccMode) == 1) {
    // Simulate disable of device 1 (second device)
    if (resources->netDev == 0) {
      resources->useBackup = 1;
      INFO(NCCL_R2CC, "R2CC_MODE=1 (RECV-CONNECT): Channel %d will use backup for device %d", 
           resources->channelId, resources->netDev);
      // R2CC DEBUG: Log listen handles state
      INFO(NCCL_R2CC, "DEBUG: Channel %d in recvProxyConnect - netListenComm=%p, netListenCommBackup=%p", 
           resources->channelId, resources->netListenComm, resources->netListenCommBackup);
    }
  }
  ncclResult_t ret = ncclSuccess;
  ncclResult_t ret2 = ncclSuccess;

  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceType, resources->netDeviceVersion, true /*isRecv*/, &resources->netDeviceHandle));
  NCCLCHECK(ncclNetGetDeviceHandle(resources->netDeviceTypeBackup, resources->netDeviceVersionBackup, true /*isRecv*/, &resources->netDeviceHandleBackup));
  // Finish connection establishment from remote peer
  // TRACE(NCCL_INIT, "resources->shared %d", resources->shared);
  if (resources->shared) {
    // Shared buffers
    struct ncclProxyProgressState* progressState = &proxyState->progressState;
    if (progressState->localPeers == NULL) {
      NCCLCHECK(ncclCalloc(&progressState->localPeers, proxyState->tpLocalnRanks));
    }
    struct ncclProxyPeer** localPeers = progressState->localPeers;
    if (localPeers[resources->tpLocalRank] == NULL) {
      NCCLCHECK(ncclCalloc(localPeers + resources->tpLocalRank, 1));
    }
    connection->proxyAppendPtr = localPeers[resources->tpLocalRank]->recv.proxyAppend + resources->channelId;

    if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
      // Connect or reuse connection for a netdev/remote rank.
      if (progressState->netComms[resources->netDev] == NULL) {
        NCCLCHECK(ncclCalloc(progressState->netComms + resources->netDev, proxyState->tpnRanks));
      }
      struct ncclSharedNetComms* comms = progressState->netComms[resources->netDev] + resources->tpRemoteProxyRank;
      if (comms->recvComm[resources->channelId] == NULL) ret = proxyState->ncclNet->accept(resources->netListenComm, comms->recvComm+resources->channelId, &resources->netDeviceHandle);
      resources->netRecvComm = comms->recvComm[resources->channelId];
      if (comms->recvComm[resources->channelId]) comms->recvRefCount[resources->channelId]++;
    } else {
      ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    }
  } else {
    // Connect to remote peer
    // TRACE(NCCL_INIT, "accept netListenComm 1 net name %s", proxyState->ncclNet->name);
    // ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    // TRACE(NCCL_INIT, "accept netListenCommBackup 2 net name %s", proxyState->ncclNet->name);
    // ret2 = proxyState->ncclNet->accept(resources->netListenCommBackup, &resources->netRecvCommBackup, &resources->netDeviceHandleBackup);
    connection->proxyAppendPtr = &connection->proxyAppend;
  }


  // TRACE(NCCL_INIT, "ret = %d", ret);
  // TRACE(NCCL_INIT, "ret2 = %d", ret2);
  // if(resources->netRecvComm == NULL && resources->netRecvCommBackup == NULL){
  //   TRACE(NCCL_INIT, "default dev %d == NULL, backup  dev %d == NULL channelId %d", resources->netDev, resources->netDevBackup, resources->channelId);
  // }

  // if(resources->netRecvComm != NULL && resources->netRecvCommBackup == NULL){
  //   TRACE(NCCL_INIT, "default dev %d done, backup  dev %d == NULL channelId %d", resources->netDev, resources->netDevBackup, resources->channelId);
  // }
  //   if(resources->netRecvComm == NULL && resources->netRecvCommBackup != NULL){
  //   TRACE(NCCL_INIT, "default dev %d == NULL, backup  dev %d done channelId %d", resources->netDev, resources->netDevBackup, resources->channelId);
  // }
  


  // NCCLCHECK(ret);
  // NCCLCHECK(ret2);
  //if (resources->netRecvComm == NULL || resources->netRecvCommBackup == NULL) {
  // if (resources->netRecvComm == NULL) {
  //   *done = 0;
  //   return ncclInProgress;
  // }

  // R2CC: Parallel accept - try both connections simultaneously
  int primaryDone = 0;
  int backupDone = 0;
  
  // Try to accept PRIMARY
  if (resources->netRecvComm == NULL) {
    if (!resources->primaryAcceptStartLogged) {
      INFO(NCCL_R2CC, "Connection: Accept START rank=%d channel=%d type=PRIMARY dev=%d listenComm=%p", 
           resources->tpRank, resources->channelId, resources->netDev, resources->netListenComm);
      resources->primaryAcceptStartLogged = 1;
    }
    // R2CC DEBUG: Log before accept
    INFO(NCCL_R2CC, "DEBUG: Channel %d calling accept for PRIMARY, listenComm=%p", 
         resources->channelId, resources->netListenComm);
    ret = proxyState->ncclNet->accept(resources->netListenComm, &resources->netRecvComm, &resources->netDeviceHandle);
    INFO(NCCL_R2CC, "DEBUG: Channel %d PRIMARY accept returned %d, recvComm=%p", 
         resources->channelId, ret, resources->netRecvComm);
    NCCLCHECK(ret);
    if (resources->netRecvComm != NULL) {
      INFO(NCCL_R2CC, "Connection: Accept COMPLETED rank=%d channel=%d type=PRIMARY dev=%d recvComm=%p from remoteRank=%d", 
           resources->tpRank, resources->channelId, resources->netDev, resources->netRecvComm, resources->tpRemoteRank);
      primaryDone = 1;
    }
  } else {
    primaryDone = 1;
  }

  // Try to accept BACKUP (parallel with primary)
  if (resources->netRecvCommBackup == NULL) {
    if (!resources->backupAcceptStartLogged) {
      INFO(NCCL_R2CC, "Connection: Accept START rank=%d channel=%d type=BACKUP dev=%d listenComm=%p", 
           resources->tpRank, resources->channelId, resources->netDevBackup, resources->netListenCommBackup);
      resources->backupAcceptStartLogged = 1;
    }
    // R2CC DEBUG: Log before backup accept
    INFO(NCCL_R2CC, "DEBUG: Channel %d calling accept for BACKUP, listenCommBackup=%p", 
         resources->channelId, resources->netListenCommBackup);
    ret2 = proxyState->ncclNet->accept(resources->netListenCommBackup, &resources->netRecvCommBackup, &resources->netDeviceHandleBackup);
    INFO(NCCL_R2CC, "DEBUG: Channel %d BACKUP accept returned %d, recvCommBackup=%p", 
         resources->channelId, ret2, resources->netRecvCommBackup);
    NCCLCHECK(ret2);
    if (resources->netRecvCommBackup != NULL) {
      INFO(NCCL_R2CC, "Connection: Accept COMPLETED rank=%d channel=%d type=BACKUP dev=%d recvComm=%p from remoteRank=%d", 
           resources->tpRank, resources->channelId, resources->netDevBackup, resources->netRecvCommBackup, resources->tpRemoteRank);
      backupDone = 1;
    }
  } else {
    backupDone = 1;
  }

  // Check if both connections are complete
  if (primaryDone && backupDone) {
    *done = 1;
    INFO(NCCL_R2CC, "PARALLEL ACCEPT: Channel %d - Both connections COMPLETE", resources->channelId);
  } else {
    *done = 0;
    const char* primaryStatus = primaryDone ? "DONE" : "ACCEPTING";
    const char* backupStatus = backupDone ? "DONE" : "ACCEPTING";
    INFO(NCCL_R2CC, "PARALLEL ACCEPT: Channel %d - PRIMARY=%s, BACKUP=%s", 
         resources->channelId, primaryStatus, backupStatus);
    return ncclInProgress;
  }

  *done = 1;
  TRACE(NCCL_INIT, "recvProxyConnect done with two comm channelId %d", resources->channelId);

  if (resources->netDeviceHandle) {
    connection->netDeviceHandle = resources->netDeviceHandle;
    connection->needsProxyProgress = connection->netDeviceHandle->needsProxyProgress;
  } else {
    connection->needsProxyProgress = 1;
  }

  if (resources->netDeviceHandleBackup) {
    connection->netDeviceHandleBackup = resources->netDeviceHandleBackup;
    connection->needsProxyProgress = connection->netDeviceHandleBackup->needsProxyProgress;
  } else {
    connection->needsProxyProgress = 1;
  }

  // R2CC: Do NOT close listen comms here - they may be needed for backup connections
  // The listen comms will be properly closed during resource cleanup

  // Create structures
  struct connectMap* map = &resources->map;
  map->sameProcess = connection->sameProcess;
  if (map->sameProcess == 0) return ncclInternalError; // We don't support remote proxy for recv
  map->shared = resources->shared;

  if (resources->shared == 0) { // Only allocate dedicated buffers for ring/tree, not for p2p
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      NCCL_NET_MAP_ADD_POINTER(map, 0, resources->useGdr, proxyState->buffSizes[p], buffs[p]);
      resources->buffSizes[p] = proxyState->buffSizes[p];
    }
  } else {
    // Get shared buffers
    int bank = resources->useGdr ? NCCL_NET_MAP_SHARED_DEVMEM : NCCL_NET_MAP_SHARED_HOSTMEM;
    struct connectMapMem* mapMem = map->mems+bank;
    NCCLCHECK(sharedNetBuffersInit(
          proxyState, resources->useGdr, resources->tpLocalRank, 1, 1, proxyState->p2pnChannels,
          &mapMem->gpuPtr, &mapMem->cpuPtr, &mapMem->size, NULL));
    resources->buffSizes[NCCL_PROTO_SIMPLE] = mapMem->size;
    NCCL_NET_MAP_ADD_POINTER(map, 1, resources->useGdr, mapMem->size, buffs[NCCL_PROTO_SIMPLE]);
  }

  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclSendMem), sendMem);
  NCCL_NET_MAP_ADD_POINTER(map, 0, 0, sizeof(struct ncclRecvMem), recvMem);

  if (proxyState->allocP2pNetLLBuffers) {
    NCCL_NET_MAP_ADD_POINTER(map, 0, 0 /*resources->useGdr*/, proxyState->buffSizes[NCCL_PROTO_LL], buffs[NCCL_PROTO_LL]);
    resources->buffSizes[NCCL_PROTO_LL] = proxyState->buffSizes[NCCL_PROTO_LL];
  }

  if (map->mems[NCCL_NET_MAP_DEVMEM].size) {
    if (resources->shared == 0) {
      if (ncclCuMemEnable()) {
        NCCLCHECK(ncclP2pAllocateShareableBuffer(map->mems[NCCL_NET_MAP_DEVMEM].size, 0, &map->mems[NCCL_NET_MAP_DEVMEM].ipcDesc,
                                                 (void**)&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr));
      } else {
        NCCLCHECK(ncclCudaCalloc(&map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr, map->mems[NCCL_NET_MAP_DEVMEM].size));
      }
      map->mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map->mems[NCCL_NET_MAP_DEVMEM].gpuPtr;
    }
  }
  NCCLCHECK(ncclCudaHostCalloc(&map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr, map->mems[NCCL_NET_MAP_HOSTMEM].size));
  map->mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = map->mems[NCCL_NET_MAP_HOSTMEM].cpuPtr;
  if (ncclGdrCopy && map->sameProcess) {
    uint64_t *cpuPtr, *gpuPtr;
    NCCLCHECK(ncclGdrCudaCalloc(&cpuPtr, &gpuPtr, 2, &resources->gdrDesc));

    if (ncclParamGdrCopySyncEnable()) {
      resources->gdcSync = cpuPtr;
      struct connectMapMem* gdcMem = map->mems+NCCL_NET_MAP_GDCMEM;
      gdcMem->cpuPtr = (char*)cpuPtr;
      gdcMem->gpuPtr = (char*)gpuPtr;
      gdcMem->size = sizeof(uint64_t);
    }
    if (ncclParamGdrCopyFlushEnable()) resources->gdcFlush = cpuPtr + 1;
  }

  resources->sendMem = (struct ncclSendMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, sendMem);
  resources->recvMem = (struct ncclRecvMem*) NCCL_NET_MAP_GET_POINTER(map, cpu, recvMem);
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffers[p] = NCCL_NET_MAP_GET_POINTER(map, cpu, buffs[p]);
    if (resources->buffers[p]) {
#if CUDA_VERSION >= 11070
      /* DMA-BUF support */
      int type = NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
      if (type == NCCL_PTR_CUDA && resources->useDmaBuf) {
        int dmabuf_fd;
        CUCHECK(cuMemGetHandleForAddressRange((void *)&dmabuf_fd, (CUdeviceptr)resources->buffers[p], resources->buffSizes[p], CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandles[p]));
        NCCLCHECK(proxyState->ncclNet->regMrDmaBuf(resources->netRecvCommBackup, resources->buffers[p], resources->buffSizes[p], type, 0ULL, dmabuf_fd, &resources->mhandlesBackup[p]));
        (void)close(dmabuf_fd);
      } else // FALL-THROUGH to nv_peermem GDR path
#endif
      {
        INFO(NCCL_R2CC, "Channel %d: Registering buffer[%d] addr=%p size=%ld with PRIMARY comm=%p (dev=%d)", 
             resources->channelId, p, resources->buffers[p], resources->buffSizes[p], resources->netRecvComm, resources->netDev);
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvComm, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandles[p]));
        INFO(NCCL_R2CC, "Channel %d: PRIMARY registration successful, mhandle[%d]=%p", resources->channelId, p, resources->mhandles[p]);
        
        INFO(NCCL_R2CC, "Channel %d: Registering buffer[%d] addr=%p size=%ld with BACKUP comm=%p (dev=%d)", 
             resources->channelId, p, resources->buffers[p], resources->buffSizes[p], resources->netRecvCommBackup, resources->netDevBackup);
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvCommBackup, resources->buffers[p], resources->buffSizes[p], NCCL_NET_MAP_DEV_MEM(map, buffs[p]) ? NCCL_PTR_CUDA : NCCL_PTR_HOST, &resources->mhandlesBackup[p]));
        INFO(NCCL_R2CC, "Channel %d: BACKUP registration successful, mhandleBackup[%d]=%p", resources->channelId, p, resources->mhandlesBackup[p]);
      }

      // Copy the mhandle dptr
      if (resources->netDeviceType != NCCL_NET_DEVICE_HOST && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netRecvComm, resources->mhandles[p], &connection->mhandles[p]));

      if (resources->netDeviceType != NCCL_NET_DEVICE_HOST && proxyState->ncclNet->getDeviceMr)
        NCCLCHECK(proxyState->ncclNet->getDeviceMr(resources->netRecvCommBackup, resources->mhandlesBackup[p], &connection->mhandlesBackup[p]));
    }
  }

  //NCCLCHECK(netDumpMap(map));
  if (respSize != sizeof(struct connectMap)) return ncclInternalError;
  memcpy(respBuff, map, sizeof(struct connectMap));
  return ncclSuccess;
}

static ncclResult_t sendProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct sendNetResources* resources = (struct sendNetResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect
    NCCLCHECK(sharedNetBuffersDestroy(proxyState, connection->tpLocalRank, 0, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    // R2CC: Deregister memory for both primary and backup paths
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->buffers[p]) {
        if (resources->netSendComm) {
          NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, resources->mhandles[p]));
        }
        if (resources->netSendCommBackup) {
          NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendCommBackup, resources->mhandlesBackup[p]));
        }
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    if (resources->map.sameProcess) {
      NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    } else {
      NCCLCHECK(ncclShmIpcClose(&mems[NCCL_NET_MAP_HOSTMEM].createDesc));
    }
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // cuMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
    if (resources->shared) {
      NCCLCHECK(sharedNetBuffersDestroy(proxyState, resources->tpLocalRank, 0, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev]+resources->tpRemoteRank;
        comms->sendRefCount[resources->channelId]--;
        if (comms->sendRefCount[resources->channelId] == 0) {
          if (comms->sendComm[resources->channelId]) {
            NCCLCHECK(proxyState->ncclNet->closeSend(comms->sendComm[resources->channelId]));
            comms->sendComm[resources->channelId] = NULL;
          }
        }
      } else {
        if (resources->netSendComm) {
          NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendComm));
        }
        if (resources->netSendCommBackup) {
          NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendCommBackup));
        }
      }
    } else {
      // R2CC: Close both primary and backup send comms
      if (resources->netSendComm) {
        NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendComm));
      }
      if (resources->netSendCommBackup) {
        NCCLCHECK(proxyState->ncclNet->closeSend(resources->netSendCommBackup));
      }
    }
  }

  // R2CC: Clear the connection's transport resources pointer
  if (resources) {
    INFO(NCCL_R2CC, "sendProxyFree: Freeing resources for channel %d", resources->channelId);
    connection->transportResources = NULL;
    free(resources);
  }
  return ncclSuccess;
}

static ncclResult_t recvProxyFree(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState) {
  struct recvNetResources* resources = (struct recvNetResources*)(connection->transportResources);
  if (connection->state == connSharedInitialized) { // NVB Preconnect
    NCCLCHECK(sharedNetBuffersDestroy(proxyState, connection->tpLocalRank, 1, connection));
    return ncclSuccess;
  }

  if (connection->state == connConnected) {
    // R2CC: Deregister memory for both primary and backup paths
    for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
      if (resources->buffers[p]) {
        if (resources->netRecvComm) {
          NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, resources->mhandles[p]));
        }
        if (resources->netRecvCommBackup) {
          NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvCommBackup, resources->mhandlesBackup[p]));
        }
      }
    }
    struct connectMapMem* mems = resources->map.mems;
    NCCLCHECK(ncclCudaHostFree(mems[NCCL_NET_MAP_HOSTMEM].cpuPtr));
    NCCLCHECK(ncclCudaFree(mems[NCCL_NET_MAP_DEVMEM].cpuPtr));
    if (!resources->map.sameProcess || ncclCuMemEnable()) {
      // cuMem API support
      if (mems[NCCL_NET_MAP_DEVMEM].size) {
        NCCLCHECK(ncclP2pFreeShareableBuffer(&mems[NCCL_NET_MAP_DEVMEM].ipcDesc));
      }
    }
    if (mems[NCCL_NET_MAP_GDCMEM].cpuPtr) NCCLCHECK(ncclGdrCudaFree(resources->gdrDesc));
    if (resources->shared) {
      NCCLCHECK(sharedNetBuffersDestroy(proxyState, resources->tpLocalRank, 1, connection));
      if (resources->maxRecvs > 1 && ncclParamNetSharedComms()) {
        struct ncclSharedNetComms* comms = proxyState->progressState.netComms[resources->netDev] + resources->tpRemoteProxyRank;
        comms->recvRefCount[resources->channelId]--;
        if (comms->recvRefCount[resources->channelId] == 0) {
          if (comms->recvComm[resources->channelId]) {
            NCCLCHECK(proxyState->ncclNet->closeRecv(comms->recvComm[resources->channelId]));
            comms->recvComm[resources->channelId] = NULL;
          }
        }
      } else {
        if (resources->netRecvComm) {
          NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvComm));
        }
        if (resources->netRecvCommBackup) {
          NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvCommBackup));
        }
      }
    } else {
      // R2CC: Close both primary and backup recv comms
      if (resources->netRecvComm) {
        NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvComm));
      }
      if (resources->netRecvCommBackup) {
        NCCLCHECK(proxyState->ncclNet->closeRecv(resources->netRecvCommBackup));
      }
    }
    
    // R2CC: Close listen comms if they still exist
    if (resources->netListenComm) {
      INFO(NCCL_R2CC, "recvProxyFree: Closing netListenComm for channel %d", resources->channelId);
      NCCLCHECK(proxyState->ncclNet->closeListen(resources->netListenComm));
      resources->netListenComm = NULL;
    }
    if (resources->netListenCommBackup) {
      INFO(NCCL_R2CC, "recvProxyFree: Closing netListenCommBackup for channel %d", resources->channelId);
      NCCLCHECK(proxyState->ncclNet->closeListen(resources->netListenCommBackup));
      resources->netListenCommBackup = NULL;
    }
  }

  // R2CC: Clear the connection's transport resources pointer
  if (resources) {
    INFO(NCCL_R2CC, "recvProxyFree: Freeing resources for channel %d", resources->channelId);
    connection->transportResources = NULL;
    free(resources);
  }
  return ncclSuccess;
}

static_assert(NCCL_STEPS <= NCCL_NET_MAX_REQUESTS, "Not enough net requests to cover for steps");
#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.


#include <thread>
#include <chrono>

#include <ctime>
#include <iomanip>
#include <sstream>
#include <mutex>

int send_total_count = 0;
int log_counter = 0;
int isend_counter=0;
int if_posted_counter=0;
int if_transmitted_counter=0;
int if_reg_counter=0;

enum R2ccProxyTraceStage {
  R2CC_PROXY_STAGE_READY = 0,
  R2CC_PROXY_STAGE_STEP_SYNC_APPLY = 1,
  R2CC_PROXY_STAGE_WAIT_STEP_SYNC = 2,
  R2CC_PROXY_STAGE_POST_GPU = 3,
  R2CC_PROXY_STAGE_WAIT_GPU_READY = 4,
  R2CC_PROXY_STAGE_ISEND_POSTED = 5,
  R2CC_PROXY_STAGE_WAIT_SEND_TEST = 6,
  R2CC_PROXY_STAGE_FAILOVER_SWITCH = 7,
  R2CC_PROXY_STAGE_IRECV_POSTED = 8,
  R2CC_PROXY_STAGE_WAIT_RECV_TEST = 9,
  R2CC_PROXY_STAGE_RECV_DONE = 10,
  R2CC_PROXY_STAGE_WAIT_FLUSH_TEST = 11,
  R2CC_PROXY_STAGE_WAIT_SENDHEAD_ACK = 12,
  R2CC_PROXY_STAGE_SUB_DONE = 13,
  R2CC_PROXY_STAGE_OP_DONE = 14
};

static const char* r2ccProxyStageName(int stage) {
  switch (stage) {
    case R2CC_PROXY_STAGE_READY: return "READY";
    case R2CC_PROXY_STAGE_STEP_SYNC_APPLY: return "STEP_SYNC_APPLY";
    case R2CC_PROXY_STAGE_WAIT_STEP_SYNC: return "WAIT_STEP_SYNC";
    case R2CC_PROXY_STAGE_POST_GPU: return "POST_GPU";
    case R2CC_PROXY_STAGE_WAIT_GPU_READY: return "WAIT_GPU_READY";
    case R2CC_PROXY_STAGE_ISEND_POSTED: return "ISEND_POSTED";
    case R2CC_PROXY_STAGE_WAIT_SEND_TEST: return "WAIT_SEND_TEST";
    case R2CC_PROXY_STAGE_FAILOVER_SWITCH: return "FAILOVER_SWITCH";
    case R2CC_PROXY_STAGE_IRECV_POSTED: return "IRECV_POSTED";
    case R2CC_PROXY_STAGE_WAIT_RECV_TEST: return "WAIT_RECV_TEST";
    case R2CC_PROXY_STAGE_RECV_DONE: return "RECV_DONE";
    case R2CC_PROXY_STAGE_WAIT_FLUSH_TEST: return "WAIT_FLUSH_TEST";
    case R2CC_PROXY_STAGE_WAIT_SENDHEAD_ACK: return "WAIT_SENDHEAD_ACK";
    case R2CC_PROXY_STAGE_SUB_DONE: return "SUB_DONE";
    case R2CC_PROXY_STAGE_OP_DONE: return "OP_DONE";
    default: return "UNKNOWN";
  }
}

static int r2ccTraceChannelFilter() {
  static int filter = []() {
    const char* env = getenv("R2CC_TRACE_CHANNEL");
    return env ? atoi(env) : -1;
  }();
  return filter;
}

static int r2ccTraceStallIters() {
  static int stallIters = []() {
    const char* env = getenv("R2CC_TRACE_STALL_ITERS");
    int v = env ? atoi(env) : 2000;
    return v > 0 ? v : 2000;
  }();
  return stallIters;
}

static int r2ccTraceTransitions() {
  static int traceTransitions = []() {
    const char* env = getenv("R2CC_TRACE_TRANSITIONS");
    return env ? atoi(env) : 0;
  }();
  return traceTransitions;
}

static inline uint64_t r2ccNowMs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)(ts.tv_nsec / 1000000ull);
}

static inline bool r2ccShouldWarnStall(uint64_t nowMs, uint64_t* startMs, uint64_t* lastWarnMs,
                                       uint64_t firstWarnMs, uint64_t repeatWarnMs) {
  if (startMs == NULL || lastWarnMs == NULL) return false;
  if (*startMs == 0) *startMs = nowMs;
  uint64_t waited = nowMs - *startMs;
  if (waited < firstWarnMs) return false;
  if (*lastWarnMs == 0 || (nowMs - *lastWarnMs) >= repeatWarnMs) {
    *lastWarnMs = nowMs;
    return true;
  }
  return false;
}

enum R2ccSendStallReason {
  R2CC_SEND_STALL_NONE = 0,
  R2CC_SEND_STALL_WAIT_RECV_TAIL = 1,
  R2CC_SEND_STALL_WAIT_GPU_READY = 2,
  R2CC_SEND_STALL_WAIT_ISEND_ALLOC = 3,
  R2CC_SEND_STALL_WAIT_SEND_TEST = 4
};

static inline const char* r2ccSendStallReasonName(int reason) {
  switch (reason) {
    case R2CC_SEND_STALL_WAIT_RECV_TAIL: return "WAIT_RECV_TAIL_OR_CONNFIFO";
    case R2CC_SEND_STALL_WAIT_GPU_READY: return "WAIT_GPU_DATA_READY";
    case R2CC_SEND_STALL_WAIT_ISEND_ALLOC: return "WAIT_ISEND_REQUEST_ALLOC";
    case R2CC_SEND_STALL_WAIT_SEND_TEST: return "WAIT_SEND_TEST_COMPLETE";
    default: return "NONE";
  }
}

static inline void r2ccSendClearStall(struct sendNetResources* resources) {
  if (resources == NULL) return;
  resources->stallReason = R2CC_SEND_STALL_NONE;
  resources->stallStartMs = 0;
  resources->stallLastWarnMs = 0;
}

static inline void r2ccSendObserveStall(struct ncclProxyArgs* args, struct ncclProxySubArgs* sub,
                                        struct sendNetResources* resources, int reason,
                                        int doneCode, int buffSlot, int connFifoSize,
                                        uint64_t recvTail, uint64_t tail) {
  if (resources == NULL || sub == NULL) return;
  if (reason == R2CC_SEND_STALL_NONE) {
    r2ccSendClearStall(resources);
    return;
  }
  uint64_t posted = (uint64_t)sub->posted;
  uint64_t transmitted = (uint64_t)sub->transmitted;
  uint64_t done = (uint64_t)sub->done;
  uint64_t nowMs = r2ccNowMs();
  bool snapshotChanged = (resources->stallReason != reason) ||
                         (resources->stallPosted != posted) ||
                         (resources->stallTransmitted != transmitted) ||
                         (resources->stallDone != done);
  if (snapshotChanged) {
    resources->stallReason = reason;
    resources->stallStartMs = nowMs;
    resources->stallLastWarnMs = 0;
    resources->stallPosted = posted;
    resources->stallTransmitted = transmitted;
    resources->stallDone = done;
  }
  if (!r2ccShouldWarnStall(nowMs, &resources->stallStartMs, &resources->stallLastWarnMs, 5000, 30000)) return;
  uint64_t waitedMs = nowMs - resources->stallStartMs;
  WARN("R2CC_STALL SEND reason=%s ch=%d conn=%d opId=%d peer=%d useBackup=%d waitMs=%" PRIu64
       " sub{posted=%" PRIu64 " transmitted=%" PRIu64 " done=%" PRIu64 " nsteps=%" PRIu64 "}"
       " detail{doneCode=%d buffSlot=%d connFifoSize=%d recvTail=%" PRIu64 " tail=%" PRIu64 " req=%p}",
       r2ccSendStallReasonName(reason), sub->channelId, resources->connIndex, args ? args->id : -1,
       resources->tpRemoteRank, resources->useBackup, waitedMs,
       posted, transmitted, done, (uint64_t)sub->nsteps,
       doneCode, buffSlot, connFifoSize, recvTail, tail,
       (buffSlot >= 0 && buffSlot < NCCL_STEPS) ? sub->requests[buffSlot] : NULL);
}

struct R2ccProxyTraceEntry {
  int stage;
  int useBackup;
  int stepSyncRequested;
  uint64_t lastPosted;
  uint64_t lastTransmitted;
  uint64_t lastDone;
  uint64_t lastNsteps;
  uint64_t stallIters;
};

static void r2ccTraceProxyState(
    const char* role, int opId, int channelId, int stage,
    const struct ncclProxySubArgs* sub, int useBackup,
    int stepSyncRequested, const char* note, bool forceLog) {
  int filter = r2ccTraceChannelFilter();
  if (filter >= 0 && channelId != filter) return;

  static const int kMaxTraceChannels = 64;
  static R2ccProxyTraceEntry sendEntry[kMaxTraceChannels];
  static R2ccProxyTraceEntry recvEntry[kMaxTraceChannels];
  static bool inited = false;
  static std::mutex traceMutex;

  if (channelId < 0 || channelId >= kMaxTraceChannels) return;

  std::lock_guard<std::mutex> lock(traceMutex);
  if (!inited) {
    for (int i = 0; i < kMaxTraceChannels; ++i) {
      sendEntry[i] = {-1, -1, -1, 0, 0, 0, 0, 0};
      recvEntry[i] = {-1, -1, -1, 0, 0, 0, 0, 0};
    }
    inited = true;
  }

  R2ccProxyTraceEntry* entry = (strcmp(role, "SEND") == 0) ? &sendEntry[channelId] : &recvEntry[channelId];
  uint64_t posted = sub ? (uint64_t)sub->posted : 0;
  uint64_t transmitted = sub ? (uint64_t)sub->transmitted : 0;
  uint64_t done = sub ? (uint64_t)sub->done : 0;
  uint64_t nsteps = sub ? (uint64_t)sub->nsteps : 0;
  int peer = sub ? sub->peer : -1;

  bool snapshotSame = (entry->stage == stage) &&
                      (entry->useBackup == useBackup) &&
                      (entry->stepSyncRequested == stepSyncRequested) &&
                      (entry->lastPosted == posted) &&
                      (entry->lastTransmitted == transmitted) &&
                      (entry->lastDone == done) &&
                      (entry->lastNsteps == nsteps);
  if (snapshotSame) {
    entry->stallIters++;
  } else {
    entry->stallIters = 0;
  }
  int stallThreshold = r2ccTraceStallIters();
  bool stallHit = snapshotSame && entry->stallIters > 0 &&
                  ((entry->stallIters % (uint64_t)stallThreshold) == 0);
  bool transitionLog = (!snapshotSame) && (r2ccTraceTransitions() != 0);
  if (!forceLog && !stallHit && !transitionLog) {
    return;
  }

  INFO(NCCL_R2CC,
       "R2CC_PROXY_STATE role=%s opId=%d channel=%d peer=%d stage=%s posted=%" PRIu64
       " transmitted=%" PRIu64 " done=%" PRIu64 " nsteps=%" PRIu64
       " useBackup=%d stepSyncRequested=%d stallIters=%" PRIu64 " note=%s",
       role, opId, channelId, peer, r2ccProxyStageName(stage), posted,
       transmitted, done, nsteps, useBackup, stepSyncRequested, entry->stallIters, note ? note : "-");

  entry->stage = stage;
  entry->useBackup = useBackup;
  entry->stepSyncRequested = stepSyncRequested;
  entry->lastPosted = posted;
  entry->lastTransmitted = transmitted;
  entry->lastDone = done;
  entry->lastNsteps = nsteps;
}

static constexpr int R2CC_FAILOVER_DIR_S2R = 0;

// Roll the send proxy back to the absolute step agreed on by the peer, then
// route subsequent network operations through the pre-established backup comm.
static inline void r2ccSendRollbackCommToAbs(struct ncclProxyArgs* args, struct sendNetResources* targetRes,
                                             uint64_t rollbackAbsStep) {
  if (targetRes == NULL) return;
  for (int s = 0; s < args->nsubs; ++s) {
    struct ncclProxySubArgs* sub = args->subs + s;
    struct sendNetResources* subRes = (struct sendNetResources*) (sub->connection->transportResources);
    if (subRes != targetRes) continue;
    uint64_t rollbackStep = 0;
    if (rollbackAbsStep > sub->base) rollbackStep = rollbackAbsStep - sub->base;
    if (rollbackStep > sub->done) rollbackStep = sub->done;
    if (rollbackStep > sub->transmitted) rollbackStep = sub->transmitted;
    for (uint64_t i = rollbackStep; i < sub->transmitted; ++i) {
      int buffSlot = (sub->base + i) % NCCL_STEPS;
      sub->requests[buffSlot] = NULL;
    }
    sub->done = rollbackStep;
    sub->transmitted = rollbackStep;
    sub->mhandle = subRes->mhandlesBackup[args->protocol];
    subRes->useBackup = 1;
  }
}

static inline ncclResult_t r2ccSendStartFailoverReq(struct ncclProxyArgs* args, struct ncclProxySubArgs* triggerSub,
                                                    struct sendNetResources* targetRes, uint64_t epochFloor,
                                                    const char* triggerReason, uint64_t triggerAbsStep, int triggerPeer) {
  if (targetRes == NULL || triggerSub == NULL) return ncclSuccess;
  if (targetRes->failoverWaitAck) return ncclSuccess;

  uint64_t nextEpoch = targetRes->failoverEpoch + 1;
  if (epochFloor > nextEpoch) nextEpoch = epochFloor;
  targetRes->failoverEpoch = nextEpoch;
  // For receiver-triggered failover, the receiver's hint is the safe replay
  // point. The sender may have completed local socket sends whose data the
  // receiver did not safely consume before the TCP path failed.
  uint64_t localDoneAbs = triggerSub->base + triggerSub->done;
  uint64_t reqAbsStep = localDoneAbs;
  if (triggerReason && strcmp(triggerReason, "recv_failover_hint") == 0 && triggerAbsStep < reqAbsStep) {
    reqAbsStep = triggerAbsStep;
  }
  targetRes->failoverReqAbsStep = reqAbsStep;
  targetRes->failoverWaitAck = 1;
  targetRes->failoverWaitStartMs = r2ccNowMs();
  targetRes->failoverWaitLastWarnMs = 0;
  targetRes->stepSyncRequested = 1;
  targetRes->stepSyncWaitIters = 0;

  r2ccSendRollbackCommToAbs(args, targetRes, targetRes->failoverReqAbsStep);
  NCCLCHECK(OobNet::Get().SendFailoverReq(targetRes->tpRemoteRank, targetRes->channelId, targetRes->connIndex,
                                          R2CC_FAILOVER_DIR_S2R, targetRes->failoverEpoch, targetRes->failoverReqAbsStep));
  OobNet::Get().ReportFailedChannel(targetRes->channelId);
  NCCLCHECK(OobNet::Get().NotifyHotRepairOnce());
  INFO(NCCL_R2CC,
       "SEND: failover request sent ch=%d conn=%d epoch=%" PRIu64 " doneAbs=%" PRIu64
       " peer=%d trigger=%s triggerAbs=%" PRIu64 " triggerPeer=%d",
       targetRes->channelId, targetRes->connIndex, targetRes->failoverEpoch,
       targetRes->failoverReqAbsStep, targetRes->tpRemoteRank,
       triggerReason ? triggerReason : "unknown", triggerAbsStep, triggerPeer);
  return ncclSuccess;
}

static inline ncclResult_t r2ccSendCheckFailoverAck(struct sendNetResources* targetRes, bool* ackApplied) {
  if (ackApplied) *ackApplied = false;
  if (targetRes == NULL || !targetRes->failoverWaitAck) return ncclSuccess;

  uint64_t ackEpoch = 0;
  uint64_t ackStep = 0;
  int ackPeer = -1;
  if (!OobNet::Get().ConsumeFailoverAck(targetRes->channelId, targetRes->connIndex, R2CC_FAILOVER_DIR_S2R,
                                        &ackEpoch, &ackStep, &ackPeer)) {
    return ncclSuccess;
  }

  if (ackEpoch < targetRes->failoverEpoch) {
    INFO(NCCL_R2CC, "SEND: ignore stale failover ack ch=%d conn=%d ackEpoch=%" PRIu64 " localEpoch=%" PRIu64,
         targetRes->channelId, targetRes->connIndex, ackEpoch, targetRes->failoverEpoch);
    return ncclSuccess;
  }

  targetRes->failoverWaitAck = 0;
  targetRes->failoverWaitStartMs = 0;
  targetRes->failoverWaitLastWarnMs = 0;
  targetRes->stepSyncRequested = 0;
  targetRes->stepSyncWaitIters = 0;
  if (ackApplied) *ackApplied = true;
  INFO(NCCL_R2CC, "SEND: failover ack applied ch=%d conn=%d epoch=%" PRIu64 " ackDoneAbs=%" PRIu64 " peer=%d",
       targetRes->channelId, targetRes->connIndex, ackEpoch, ackStep, ackPeer);
  return ncclSuccess;
}

static inline ncclResult_t r2ccSendCheckFailoverHint(struct ncclProxyArgs* args, struct ncclProxySubArgs* sub,
                                                     struct sendNetResources* resources, bool* triggered) {
  if (triggered) *triggered = false;
  if (args == NULL || sub == NULL || resources == NULL) return ncclSuccess;
  if (resources->failoverWaitAck) return ncclSuccess;

  uint64_t hintEpoch = 0;
  uint64_t hintRecvAbs = 0;
  int hintPeer = -1;
  if (!OobNet::Get().ConsumeFailoverHint(resources->channelId, resources->connIndex, R2CC_FAILOVER_DIR_S2R,
                                         &hintEpoch, &hintRecvAbs, &hintPeer)) {
    return ncclSuccess;
  }

  if (hintPeer >= 0 && hintPeer != resources->tpRemoteRank) {
    WARN("SEND: ignore FAILOVER_HINT from unexpected peer ch=%d conn=%d expectedPeer=%d gotPeer=%d",
         resources->channelId, resources->connIndex, resources->tpRemoteRank, hintPeer);
    return ncclSuccess;
  }

  if (hintEpoch != 0 && hintEpoch <= resources->failoverEpoch) {
    return ncclSuccess;
  }

  INFO(NCCL_R2CC,
       "SEND: received FAILOVER_HINT ch=%d conn=%d hintEpoch=%" PRIu64 " recvAbs=%" PRIu64 " peer=%d",
       resources->channelId, resources->connIndex, hintEpoch, hintRecvAbs,
       (hintPeer >= 0) ? hintPeer : resources->tpRemoteRank);
  NCCLCHECK(r2ccSendStartFailoverReq(args, sub, resources, hintEpoch, "recv_failover_hint", hintRecvAbs, hintPeer));
  if (triggered) *triggered = true;
  return ncclSuccess;
}

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // During proxy shutdown, force-complete outstanding ops to avoid teardown races.
  if (proxyState->progressState.stop ||
      (proxyState->abortFlag && __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) != 0)) {
    args->done = args->nsubs;
    args->state = ncclProxyOpNone;
    args->idle = 1;
    return ncclSuccess;
  }

  // for (int s=0; s<args->nsubs; s++) {
  //   struct ncclProxySubArgs* sub = args->subs+s;
  //   struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
  //   int change = 0;
  //   //proxyState->ncclNet->checkSwitchToBackup(resources->netSendComm, &change);
  //   //TRACE(NCCL_NET, "netSendComm change %d", change);
  //   proxyState->ncclNet->checkSwitchToBackup(resources->netSendCommBackup, &change);
  //   TRACE(NCCL_NET, "netSendCommBackup change %d", change);
  //   exit(0);
  // }




  if (args->state == ncclProxyOpReady) {
    // Add timestamp for send start
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    if (args->subs && args->nsubs > 0) {
      INFO(NCCL_R2CC, "[TIMESTAMP] SendProxy START: channel=%d peer=%d time=%ld.%09ld", 
           args->subs[0].channelId, args->subs[0].peer, start_time.tv_sec, start_time.tv_nsec);
    }
    send_total_count++;
    args->id = send_total_count;  
    // TRACE(NCCL_NET, "sendproxyprogress: [%s] id=%d 1. ncclProxyOpReady", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendNetResources* resources =  (struct sendNetResources*) (sub->connection->transportResources);

      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->transmitted = sub->done = 0;
      resources->stepSyncWaitIters = 0;
      if (!resources->failoverWaitAck) {
        resources->failoverWaitStartMs = 0;
        resources->failoverWaitLastWarnMs = 0;
      }
      r2ccSendClearStall(resources);
      ncclProfilerStartSendProxyOpEvent(s, args);


      static int forceSendBackupChannels = -1;
      if (forceSendBackupChannels == -1) {
        const char* env = getenv("NCCL_FORCE_BACKUP_CHANNELS");
        forceSendBackupChannels = env ? atoi(env) : 0;
      }
      
      if (forceSendBackupChannels) {
        if(sub->channelId == 0 || sub->channelId == 8){
          resources->useBackup = 1;
        }
      }


      if (sub->reg && sub->nbytes > 0) {
        // Register with both comms for consistency
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netSendComm, sub->recvbuff, sub->nbytes, NCCL_PTR_CUDA, &sub->mhandle));
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netSendCommBackup, sub->recvbuff, sub->nbytes, NCCL_PTR_CUDA, &sub->mhandleBackup));
        INFO(NCCL_R2CC, "SEND: Channel %d registered memory with both PRIMARY and BACKUP comms, buffer=%p, size=%ld", 
             sub->channelId, sub->recvbuff, sub->nbytes);
      } else {
        // For pre-registered buffers, copy both handles from resources  
        sub->mhandle = resources->mhandles[args->protocol];
        sub->mhandleBackup = resources->mhandlesBackup[args->protocol];
      }
      r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_READY, sub,
                          resources->useBackup, resources->stepSyncRequested,
                          "op_ready_init", true);
    }
    struct ncclProxySubArgs* sub = args->subs+0;
    struct sendNetResources* resources =  (struct sendNetResources*) (sub->connection->transportResources);
    TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: init ncclProxyOpReady", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    // Poll OOB mailbox for receiver hints and failover ACKs per backup context.
    OobNet& oob = OobNet::Get();
    oob.PollHotRepair();
    const uint64_t nowMs = r2ccNowMs();
    bool anyStepSyncWait = false;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      bool seenComm = false;
      for (int p = 0; p < s; ++p) {
        struct sendNetResources* otherRes = (struct sendNetResources*) (args->subs[p].connection->transportResources);
        if (otherRes == resources) {
          seenComm = true;
          break;
        }
      }
      if (seenComm) continue;

      bool hintTriggered = false;
      NCCLCHECK(r2ccSendCheckFailoverHint(args, sub, resources, &hintTriggered));
      if (hintTriggered) args->idle = 0;

      bool ackApplied = false;
      NCCLCHECK(r2ccSendCheckFailoverAck(resources, &ackApplied));
      if (ackApplied) {
        INFO(NCCL_R2CC, "SEND: failover ACK complete ch=%d conn=%d epoch=%" PRIu64,
             resources->channelId, resources->connIndex, resources->failoverEpoch);
      }

      resources->stepSyncRequested = resources->failoverWaitAck ? 1 : 0;
      if (resources->failoverWaitAck) {
        if (r2ccShouldWarnStall(nowMs, &resources->failoverWaitStartMs, &resources->failoverWaitLastWarnMs, 5000, 30000)) {
          uint64_t waitedMs = nowMs - resources->failoverWaitStartMs;
          WARN("R2CC_STALL SEND waiting FAILOVER_ACK >5s ch=%d conn=%d epoch=%" PRIu64
               " peer=%d reqDoneAbs=%" PRIu64 " waitMs=%" PRIu64
               " sub{posted=%" PRIu64 " transmitted=%" PRIu64 " done=%" PRIu64 " nsteps=%" PRIu64 "} useBackup=%d",
               resources->channelId, resources->connIndex, resources->failoverEpoch,
               resources->tpRemoteRank, resources->failoverReqAbsStep, waitedMs,
               (uint64_t)sub->posted, (uint64_t)sub->transmitted, (uint64_t)sub->done, (uint64_t)sub->nsteps,
               resources->useBackup);
        }
        anyStepSyncWait = true;
        resources->stepSyncWaitIters++;
        if ((resources->stepSyncWaitIters % 5000) == 0) {
          NCCLCHECK(OobNet::Get().SendFailoverReq(resources->tpRemoteRank, resources->channelId, resources->connIndex,
                                                  R2CC_FAILOVER_DIR_S2R, resources->failoverEpoch, resources->failoverReqAbsStep));
          INFO(NCCL_R2CC, "SEND: re-send FAILOVER_REQ ch=%d conn=%d epoch=%" PRIu64
               " doneAbs=%" PRIu64 " peer=%d waitIters=%d",
               resources->channelId, resources->connIndex, resources->failoverEpoch,
               resources->failoverReqAbsStep, resources->tpRemoteRank, resources->stepSyncWaitIters);
        }
        r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_WAIT_STEP_SYNC, sub,
                            resources->useBackup, resources->stepSyncRequested,
                            "waiting_failover_ack", false);
      }
    }
    if (anyStepSyncWait) {
      args->idle = 1;
      return ncclSuccess;
    }


    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (sub->done == sub->nsteps) continue;
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      if (resources->stepSyncRequested) continue;
      volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
      int stepSize = resources->buffSizes[p] / NCCL_STEPS;
      char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
      // Post buffers to the GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + maxDepth) {
        if_posted_counter++;
        if(if_posted_counter %1000007==0){
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d, if_posted_counter++", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
        }
        // TRACE(NCCL_NET, "sendproxyprogress: [%s] id=%d 2. Post buffers to shared buffer", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
        ncclProfilerStartSendProxyStepEvents(s, args, sub->posted, sub->posted+args->sliceSteps);
        int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
        if (resources->shared) {
          if (!sub->reg) {
            int sharedBuffSlot = sub->posted%maxDepth;
            int offset;
            NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot*args->nsubs+s, &offset, NULL));
            resources->recvMem->connFifo[buffSlot].offset = offset;
            __sync_synchronize();
          }
          volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
          sub->posted += args->sliceSteps;
          // Only post one credit for registered buffer
          if (sub->reg == 0 || sub->posted == args->sliceSteps) *sendHead = sub->base + sub->posted - NCCL_STEPS;
          if (resources->gdcSync) wc_store_fence(); // Flush out WC write
        } else sub->posted += args->sliceSteps;
        ncclProfilerRecordProxyOpEventState(s, args, sub->posted, sub->transSize, ncclProfilerProxyOpSendPosted);
        ncclProfilerRecordProxyStepEventStates(s, args, sub->posted-args->sliceSteps, sub->posted, ncclProfilerProxyStepSendGPUWait);
        r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_POST_GPU, sub,
                            resources->useBackup, resources->stepSyncRequested,
                            "posted_to_gpu_fifo", false);
        r2ccSendClearStall(resources);
        args->idle = 0;
        continue;
      }
      // Check whether we received data from the GPU and send it to the network
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        // TRACE(NCCL_NET, "sendproxyprogress: [%s] id=%d 2. iSend it to the network", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile uint64_t* recvTail = &resources->recvMem->tail;
        uint64_t tail = sub->base + (sub->reg ? 0 : sub->transmitted);

        if_transmitted_counter++;
        if(if_transmitted_counter %1000007==0){
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d, if_transmitted_counter++", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
          int a = (sub->reg || connFifo[buffSlot].size != -1);
          int b = ((*recvTail > tail) || p == NCCL_PROTO_LL);

          TRACE(NCCL_NET, "sub->reg=%d, connFifo[buffSlot].size=%ld, (*recvTail > tail)=%d, (p == NCCL_PROTO_LL)=%d, a=%d, b=%d, done=%ld, trasnmitted=%ld, recvTail=%ld", sub->reg, connFifo[buffSlot].size, (*recvTail > tail), (p == NCCL_PROTO_LL), a, b, sub->done, sub->transmitted, (*recvTail));
        }

        ssize_t connFifoSize = connFifo[buffSlot].size;
        // During backup replay, the sender may revisit a FIFO slot whose
        // primary-path size metadata was already consumed. If the receiver has
        // advertised credit for the step, use the normal slice size so the
        // replay can be posted on the backup connection.
        if (resources->useBackup && connFifoSize == -1 && ((*recvTail > tail) || p == NCCL_PROTO_LL)) {
          connFifoSize = stepSize * args->sliceSteps;
          INFO(NCCL_R2CC,
               "SEND: backup replay using fallback connFifo size ch=%d buffSlot=%d size=%ld recvTail=%ld tail=%ld",
               sub->channelId, buffSlot, connFifoSize, *recvTail, tail);
        }
        if ((sub->reg || connFifoSize != -1) && ((*recvTail > tail) || p == NCCL_PROTO_LL)) {

          if_reg_counter++;
          if(if_reg_counter %1000007==0){
            TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d, if_reg_counter++", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
          }

          // We have something to receive, let's check if it's completely ready.
          int size = sub->reg ? std::min(MAX_NET_SIZE, sub->nbytes) : connFifoSize;
          bool shared = (p == NCCL_PROTO_SIMPLE) && resources->shared;
          char* buff = shared ? localBuff+connFifo[buffSlot].offset : localBuff+buffSlot*stepSize;
          int ready = 1;
          if (p == NCCL_PROTO_LL128) {
            ready = resources->useGdr;
            if (!ready) {
              // When data is in sysmem, we need to wait until all flags are correct since the GPU only
              // called threadfence()
              uint64_t flag = sub->base+sub->transmitted+1;
              int nFifoLines = DIVUP(connFifo[buffSlot].size, sizeof(uint64_t)*NCCL_LL128_LINEELEMS);
              volatile uint64_t* lines = (volatile uint64_t*)buff;
              ready = 1;
              for (int i=0; i<nFifoLines; i++) {
                if (lines[i*NCCL_LL128_LINEELEMS+NCCL_LL128_DATAELEMS] != flag) { ready = 0; break; }
              }
            }
          } else if (p == NCCL_PROTO_LL) {
            uint32_t flag = NCCL_LL_FLAG(sub->base+sub->transmitted+1);
            int nFifoLines = DIVUP(size, sizeof(union ncclLLFifoLine));
            union ncclLLFifoLine* lines = (union ncclLLFifoLine*)buff;
            for (int i=0; i<nFifoLines; i++) {
              volatile uint32_t *f1 = &lines[i].flag1;
              volatile uint32_t *f2 = &lines[i].flag2;
              if (f1[0] != flag || f2[0] != flag) { ready = 0; break; }
            }
          } else if (p == NCCL_PROTO_SIMPLE && resources->shared) {
            buff = sub->reg ? (char*)sub->recvbuff : localBuff+resources->recvMem->connFifo[buffSlot].offset;
          }
          if (ready) {
            ncclProfilerRecordProxyOpEventState(s, args, sub->transmitted + args->sliceSteps, sub->transSize, ncclProfilerProxyOpSendRemFifoWait);
            // Data is ready, try to send.
            // Coverity complains about the size here as pointing to an out-of-scope temporary.  Which is nonsense,
            // since size is a plain integer.
            // coverity[use_invalid:FALSE]
          

            // if(sub->channelId%2==0)
            //  NCCLCHECK(proxyState->ncclNet->isend(resources->netSendComm, buff, size, resources->tpRank, sub->mhandle, sub->requests+buffSlot));
            //else
            // std::this_thread::sleep_for(std::chrono::milliseconds(10));
              // if(resources->useBackup)
              //   TRACE(NCCL_NET, "sendProxy [%ld/%d] prepare to send, req %p, size %d, proto %d, myRank %d, channelId %d through backupComm", sub->transmitted, buffSlot, sub->requests[buffSlot], size, p, proxyState->tpRank, sub->channelId);
              // else
              //   TRACE(NCCL_NET, "sendProxy [%ld/%d] prepare to send, req %p, size %d, proto %d, myRank %d, channelId %d", sub->transmitted, buffSlot, sub->requests[buffSlot], size, p, proxyState->tpRank, sub->channelId);
      
            isend_counter++;
            if(isend_counter %1000007==0){
              TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: do isend", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
            }
            // Log which comm is being used for isend (only when MODE1 subsystem is enabled)
            if (resources->useBackup) {
              INFO(NCCL_MODE1, "SEND: Channel %d using BACKUP comm for isend, size=%d", sub->channelId, size);
            }
            INFO(NCCL_R2CC, "SEND: Calling isend for channel=%d, buffSlot=%d, useBackup=%d", sub->channelId, buffSlot, resources->useBackup);
            void* mhandleToUse = resources->useBackup ? sub->mhandleBackup : sub->mhandle;
            NCCLCHECK(proxyState->ncclNet->isend(resources->useBackup ? resources->netSendCommBackup : resources->netSendComm , buff, size, resources->tpRank, mhandleToUse, sub->requests+buffSlot));
            
            if (sub->requests[buffSlot] != NULL) {
              INFO(NCCL_R2CC, "SEND: isend allocated request %p for channel=%d, buffSlot=%d", sub->requests[buffSlot], sub->channelId, buffSlot);
              TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: allocate request success", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
              proxyState->ncclNet->setRequestChannel(&(sub->requests[buffSlot]), sub->channelId);
              proxyState->ncclNet->setRequestId(&(sub->requests[buffSlot]), args->id);
              proxyState->ncclNet->setRequestComm(&(sub->requests[buffSlot]), resources->useBackup ? (void*)(resources->netSendCommBackup) : (void*)(resources->netSendComm));
              proxyState->ncclNet->setRequestStep(&(sub->requests[buffSlot]), sub->base+sub->transmitted);
              proxyState->ncclNet->setRequestOperation(&(sub->requests[buffSlot]), 2);
              r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_ISEND_POSTED, sub,
                                  resources->useBackup, resources->stepSyncRequested,
                                  "isend_request_allocated", false);
              sub->transmitted += args->sliceSteps;
              ncclProfilerRecordProxyOpEventState(s, args, sub->transmitted, sub->transSize, ncclProfilerProxyOpSendTransmitted);
              ncclProfilerRecordProxyStepEventStates(s, args, sub->transmitted-args->sliceSteps, sub->transmitted, ncclProfilerProxyStepSendWait);
              sub->transSize += size;
              r2ccSendClearStall(resources);
              args->idle = 0;
              continue;
            }
            else{
              log_counter++;
              if(log_counter %1000007==0){
                TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: allocate request failed", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
              }
              r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_WAIT_SEND_TEST, sub,
                                  resources->useBackup, resources->stepSyncRequested,
                                  "isend_request_not_allocated", false);
              r2ccSendObserveStall(args, sub, resources, R2CC_SEND_STALL_WAIT_ISEND_ALLOC,
                                   0, buffSlot, connFifo[buffSlot].size, *recvTail, tail);
            }
          } else {
            r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_WAIT_GPU_READY, sub,
                                resources->useBackup, resources->stepSyncRequested,
                                "gpu_data_not_ready", false);
            r2ccSendObserveStall(args, sub, resources, R2CC_SEND_STALL_WAIT_GPU_READY,
                                 0, buffSlot, connFifo[buffSlot].size, *recvTail, tail);
          }
        } else {
          r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_WAIT_GPU_READY, sub,
                              resources->useBackup, resources->stepSyncRequested,
                              "waiting_recv_tail_or_connfifo", false);
          r2ccSendObserveStall(args, sub, resources, R2CC_SEND_STALL_WAIT_RECV_TAIL,
                               0, buffSlot, connFifo[buffSlot].size, *recvTail, tail);
        }
      }
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        //TRACE(NCCL_NET, "sendproxyprogress: [%s] id=%d 3. Check whether the network has completed some send operations.", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
        int done;
        int size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_WAIT_SEND_TEST, sub,
                            resources->useBackup, resources->stepSyncRequested,
                            "poll_send_request", false);
        INFO(NCCL_R2CC, "SEND: Testing request %p for channel=%d, buffSlot=%d", sub->requests[buffSlot], sub->channelId, buffSlot);
        NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));
        INFO(NCCL_R2CC, "SEND: Test result done=%d for request %p, channel=%d", done, sub->requests[buffSlot], sub->channelId);
        if (done == -1){
          // A local send-side socket error is not enough to choose a safe
          // replay point. Wait for the receiver's FAILOVER_HINT, which carries
          // the receiver's last safe absolute step.
          INFO(NCCL_R2CC, "SEND: test returned -1, channel=%d, useBackup=%d; waiting for receiver FAILOVER_HINT",
               sub->channelId, resources->useBackup);
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: test done=-1", args->id, sub->channelId, sub->base+sub->done, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
          r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_FAILOVER_SWITCH, sub,
                              resources->useBackup, resources->stepSyncRequested,
                              "test_done_minus1_wait_recv_hint", true);
          r2ccSendObserveStall(args, sub, resources, R2CC_SEND_STALL_WAIT_SEND_TEST,
                               done, buffSlot, -1, 0, 0);
          break;
        }
        if (done) {
          // Add step completion timestamp for accurate measurement
          struct timespec step_time;
          clock_gettime(CLOCK_MONOTONIC, &step_time);
          INFO(NCCL_R2CC, "[TIMESTAMP] Channel %d Step %ld COMPLETE: time=%ld.%09ld",
               sub->channelId, sub->base+sub->done, step_time.tv_sec, step_time.tv_nsec);
          
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: test done=1", args->id, sub->channelId, sub->base+sub->done, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
          if (sub->reg) {
            if (size < sub->nbytes) {
              sub->recvbuff += size;
              sub->nbytes -= size;
              // Do one more step (at least)
              sub->nsteps++;
            } else {
              // Signal the GPU the send is complete and it can return.
              connFifo[sub->base%NCCL_STEPS].size = -1;
            }
          }
          // Make sure size is reset to -1 before we update the head.
          if (sub->reg == 0) connFifo[buffSlot].size = -1;
          __sync_synchronize();
          
          // Add request completion timestamp
          struct timespec req_time;
          clock_gettime(CLOCK_MONOTONIC, &req_time);
          INFO(NCCL_R2CC, "[TIMESTAMP] Channel %d Request [%ld/%d] DONE: time=%ld.%09ld",
               sub->channelId, sub->done, buffSlot, req_time.tv_sec, req_time.tv_nsec);
          
          TRACE(NCCL_NET, "sendProxy [%ld/%d] request %p done", sub->done, buffSlot, sub->requests[buffSlot]);
          sub->done += args->sliceSteps;
          ncclProfilerStopProxyStepEvents(s, args, sub->done-args->sliceSteps, sub->done);
          ncclProfilerRecordProxyOpEventState(s, args, sub->done, sub->transSize, ncclProfilerProxyOpSendDone);

          if (resources->shared == 0) {
            volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
            if (sub->reg) {
              // We may have added more net steps, but reg operations only have a single step w.r.t. the GPU.
              if (sub->done == sub->nsteps) *sendHead = sub->base + args->sliceSteps;
            } else {
              *sendHead = sub->base + sub->done;
            }
            if (resources->gdcSync) wc_store_fence(); // Flush out WC write
          }
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            // Add timestamp for channel send completion (before sync)
            struct timespec complete_time;
            clock_gettime(CLOCK_MONOTONIC, &complete_time);
            INFO(NCCL_R2CC, "[TIMESTAMP] Channel %d SEND_DONE: time=%ld.%09ld", 
                 sub->channelId, complete_time.tv_sec, complete_time.tv_nsec);
            if (sub->reg && sub->nbytes > 0) {
              // Deregister from both comms
              NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendComm, sub->mhandle));
              NCCLCHECK(proxyState->ncclNet->deregMr(resources->netSendCommBackup, sub->mhandleBackup));
              INFO(NCCL_R2CC, "SEND: Channel %d deregistered memory from both PRIMARY and BACKUP comms", sub->channelId);
            }
            r2ccTraceProxyState("SEND", args->id, sub->channelId, R2CC_PROXY_STAGE_SUB_DONE, sub,
                                resources->useBackup, resources->stepSyncRequested,
                                "sub_all_steps_done", true);
            args->done++;
          }
        }
        else {
          r2ccSendObserveStall(args, sub, resources, R2CC_SEND_STALL_WAIT_SEND_TEST,
                               done, buffSlot, -1, 0, 0);
        }
      }
    }
    if (args->done == args->nsubs) {
      sendNetResources* resources = (struct sendNetResources*) ((args->subs+0)->connection->transportResources);
      struct ncclProxySubArgs* sub0 = args->subs + 0;
      TRACE(NCCL_NET, "id=%d, channel=%d, useBackup=%d, comm=%p, rank=%d, remoteRank=%d: args done", args->id, (args->subs+0)->channelId, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
      r2ccTraceProxyState("SEND", args->id, sub0->channelId, R2CC_PROXY_STAGE_OP_DONE, sub0,
                          resources->useBackup, resources->stepSyncRequested,
                          "proxy_op_done", true);
      for (int s=0; s<args->nsubs; s++) {
        ncclProfilerStopProxyOpEvent(s, args);
      }
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

int recv_total_count = 0;

static inline void r2ccForceUngroup(struct ncclProxyArgs* args) {
  for (int i = 0; i < args->nsubs; ++i) args->subs[i].groupSize = 1;
}

static inline void r2ccRecvRollbackCommToAbs(struct ncclProxyArgs* args, struct recvNetResources* targetRes, uint64_t senderDoneAbs, uint64_t* appliedDoneAbs) {
  if (targetRes == NULL) return;
  uint64_t applied = senderDoneAbs;
  bool appliedSet = false;
  for (int s = 0; s < args->nsubs; ++s) {
    struct ncclProxySubArgs* sub = args->subs + s;
    struct recvNetResources* subRes = (struct recvNetResources*) (sub->connection->transportResources);
    if (subRes != targetRes) continue;
    uint64_t rollbackStep = 0;
    if (senderDoneAbs > sub->base) rollbackStep = senderDoneAbs - sub->base;
    // Sender is authoritative, but never fast-forward local receive state.
    if (rollbackStep > sub->done) rollbackStep = sub->done;
    if (rollbackStep > sub->posted) rollbackStep = sub->posted;
    for (int i = 0; i < NCCL_STEPS; ++i) {
      sub->requests[i] = NULL;
      sub->recvRequestsCache[i] = NULL;
    }
    sub->recvRequestsSubCount = 0;
    sub->posted = rollbackStep;
    sub->received = rollbackStep;
    sub->transmitted = rollbackStep;
    sub->done = rollbackStep;
    sub->mhandle = subRes->mhandlesBackup[args->protocol];
    subRes->useBackup = 1;
    subRes->waitFailoverReq = 0;
    subRes->waitFailoverStartMs = 0;
    subRes->waitFailoverLastWarnMs = 0;
    subRes->waitFailoverHintEpoch = 0;
    subRes->waitFailoverHintAbsStep = 0;
    subRes->waitFailoverHintLastSendMs = 0;
    subRes->waitFailoverHintSendCount = 0;

    uint64_t subAppliedAbs = sub->base + rollbackStep;
    if (!appliedSet || subAppliedAbs < applied) {
      applied = subAppliedAbs;
      appliedSet = true;
    }
  }
  if (appliedDoneAbs) *appliedDoneAbs = applied;
}

static inline void r2ccWarnRecvWaitFailoverReq(struct ncclProxySubArgs* sub, struct recvNetResources* resources) {
  if (sub == NULL || resources == NULL) return;
  uint64_t nowMs = r2ccNowMs();
  if (!r2ccShouldWarnStall(nowMs, &resources->waitFailoverStartMs, &resources->waitFailoverLastWarnMs, 5000, 30000)) return;
  uint64_t waitedMs = nowMs - resources->waitFailoverStartMs;
  WARN("R2CC_STALL RECV waiting FAILOVER_REQ >5s ch=%d conn=%d epoch=%" PRIu64
       " peer=%d waitMs=%" PRIu64
       " sub{posted=%" PRIu64 " received=%" PRIu64 " transmitted=%" PRIu64 " done=%" PRIu64 " nsteps=%" PRIu64 "} useBackup=%d",
       sub->channelId, resources->connIndex, resources->lastFailoverEpoch,
       resources->tpRemoteRank, waitedMs,
       (uint64_t)sub->posted, (uint64_t)sub->received, (uint64_t)sub->transmitted,
       (uint64_t)sub->done, (uint64_t)sub->nsteps, resources->useBackup);
}

static inline uint64_t r2ccFailoverHintResendMs() {
  static int ms = -2;
  if (ms == -2) {
    const char* env = getenv("NCCL_R2CC_FAILOVER_HINT_RESEND_MS");
    ms = env ? atoi(env) : 500;
    if (ms < 50) ms = 50;
  }
  return (uint64_t)ms;
}

static inline ncclResult_t r2ccRecvSendFailoverHint(struct ncclProxySubArgs* sub, struct recvNetResources* resources,
                                                     uint64_t epochHint, uint64_t recvAbsStep, bool isResend) {
  if (sub == NULL || resources == NULL) return ncclSuccess;
  NCCLCHECK(OobNet::Get().SendFailoverHint(resources->tpRemoteRank, sub->channelId, resources->connIndex,
                                           R2CC_FAILOVER_DIR_S2R, epochHint, recvAbsStep));
  resources->waitFailoverHintEpoch = epochHint;
  resources->waitFailoverHintAbsStep = recvAbsStep;
  resources->waitFailoverHintLastSendMs = r2ccNowMs();
  resources->waitFailoverHintSendCount += 1;
  if (resources->waitFailoverHintSendCount == 1 || (resources->waitFailoverHintSendCount % 20) == 0) {
    INFO(NCCL_R2CC,
         "RECV: sent FAILOVER_HINT ch=%d conn=%d epochHint=%" PRIu64 " recvAbs=%" PRIu64 " peer=%d mode=%s count=%" PRIu64,
         sub->channelId, resources->connIndex, epochHint, recvAbsStep, resources->tpRemoteRank,
         isResend ? "resend" : "initial", resources->waitFailoverHintSendCount);
  }
  return ncclSuccess;
}

static inline ncclResult_t r2ccRecvHandleWaitFailoverReq(struct ncclProxySubArgs* sub, struct recvNetResources* resources) {
  if (sub == NULL || resources == NULL || !resources->waitFailoverReq) return ncclSuccess;

  const uint64_t nowMs = r2ccNowMs();
  const uint64_t resendEveryMs = r2ccFailoverHintResendMs();
  if (resources->waitFailoverHintEpoch != 0 &&
      (resources->waitFailoverHintLastSendMs == 0 || nowMs - resources->waitFailoverHintLastSendMs >= resendEveryMs)) {
    NCCLCHECK(r2ccRecvSendFailoverHint(sub, resources, resources->waitFailoverHintEpoch,
                                       resources->waitFailoverHintAbsStep, true));
  }

  r2ccWarnRecvWaitFailoverReq(sub, resources);

  const int waitMaxMs = ncclParamR2CCFailoverWaitMaxMs();
  if (waitMaxMs > 0 && resources->waitFailoverStartMs != 0 &&
      nowMs - resources->waitFailoverStartMs >= (uint64_t)waitMaxMs) {
    WARN("R2CC_RECV wait_failover_req exceeded max wait ch=%d conn=%d epoch=%" PRIu64
         " waitMs=%" PRIu64 " maxMs=%d",
         sub->channelId, resources->connIndex, resources->lastFailoverEpoch,
         nowMs - resources->waitFailoverStartMs, waitMaxMs);
    return ncclRemoteError;
  }
  return ncclSuccess;
}

static inline ncclResult_t r2ccRecvApplyPendingFailoverReq(struct ncclProxyArgs* args, struct ncclProxySubArgs* sub, bool* applied) {
  if (applied) *applied = false;
  if (sub == NULL) return ncclSuccess;

  struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
  if (resources == NULL) return ncclSuccess;

  uint64_t reqEpoch = 0;
  uint64_t reqDoneAbs = 0;
  int reqPeer = -1;
  if (!OobNet::Get().ConsumeFailoverReq(sub->channelId, resources->connIndex, R2CC_FAILOVER_DIR_S2R,
                                        &reqEpoch, &reqDoneAbs, &reqPeer)) {
    return ncclSuccess;
  }

  int ackPeer = (reqPeer >= 0) ? reqPeer : resources->tpRemoteRank;
  if (reqEpoch < resources->lastFailoverEpoch) {
    INFO(NCCL_R2CC, "RECV: ignore stale failover req ch=%d conn=%d epoch=%" PRIu64 " localEpoch=%" PRIu64,
         sub->channelId, resources->connIndex, reqEpoch, resources->lastFailoverEpoch);
    return ncclSuccess;
  }
  if (reqEpoch == resources->lastFailoverEpoch) {
    NCCLCHECK(OobNet::Get().SendFailoverAck(ackPeer, sub->channelId, resources->connIndex, R2CC_FAILOVER_DIR_S2R,
                                            reqEpoch, reqDoneAbs));
    INFO(NCCL_R2CC, "RECV: duplicate failover req re-acked ch=%d conn=%d epoch=%" PRIu64,
         sub->channelId, resources->connIndex, reqEpoch);
    return ncclSuccess;
  }

  resources->lastFailoverEpoch = reqEpoch;
  uint64_t appliedDoneAbs = reqDoneAbs;
  r2ccRecvRollbackCommToAbs(args, resources, reqDoneAbs, &appliedDoneAbs);
  resources->waitFailoverReq = 0;
  r2ccForceUngroup(args);
  NCCLCHECK(OobNet::Get().SendFailoverAck(ackPeer, sub->channelId, resources->connIndex, R2CC_FAILOVER_DIR_S2R,
                                          reqEpoch, appliedDoneAbs));
  INFO(NCCL_R2CC, "RECV: failover req applied ch=%d conn=%d epoch=%" PRIu64
       " senderDoneAbs=%" PRIu64 " appliedDoneAbs=%" PRIu64 " peer=%d",
       sub->channelId, resources->connIndex, reqEpoch, reqDoneAbs, appliedDoneAbs, ackPeer);
  if (applied) *applied = true;
  return ncclSuccess;
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  // During proxy shutdown, force-complete outstanding ops to avoid teardown races.
  if (proxyState->progressState.stop ||
      (proxyState->abortFlag && __atomic_load_n(proxyState->abortFlag, __ATOMIC_ACQUIRE) != 0)) {
    args->done = args->nsubs;
    args->state = ncclProxyOpNone;
    args->idle = 1;
    return ncclSuccess;
  }
  
  if (args->state == ncclProxyOpReady) {
    // Initialize subs and group them by same recvComm.


    recv_total_count++;
    args->id = recv_total_count;  
    // TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d 1. ncclProxyOpReady", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
    void* recvComm;
    int groupSize = 0;
    int maxRecvs = 1;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (groupSize == maxRecvs) {
        groupSize = 0;
      } else if (s>0) { // Find next sub with the same recvComm
        int next;
        for (next=s; next<args->nsubs; next++) {
          struct recvNetResources* nextRes = (struct recvNetResources*) (args->subs[next].connection->transportResources);
          void* nextComm = nextRes->useBackup ? nextRes->netRecvCommBackup : nextRes->netRecvComm;
          if (nextComm == recvComm) break;
        }
        if (next == args->nsubs) { // Not found
          groupSize = 0;
        } else if (s != next) { // We found a sub later with the same recvComm ; swap subs
          struct ncclProxySubArgs temp;
          memcpy(&temp, sub, sizeof(struct ncclProxySubArgs));
          memcpy(sub, args->subs+next, sizeof(struct ncclProxySubArgs));
          memcpy(args->subs+next, &temp, sizeof(struct ncclProxySubArgs));
        }
      }
      groupSize++;
      struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
      maxRecvs = resources->useBackup ? resources->maxRecvsBackup : resources->maxRecvs;
      recvComm = resources->useBackup ? resources->netRecvCommBackup : resources->netRecvComm;

      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->step, args->chunkSteps);
      // Set step base for next op
      resources->step = sub->base + sub->nsteps;
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
      resources->waitFailoverReq = 0;
      resources->waitFailoverStartMs = 0;
      resources->waitFailoverLastWarnMs = 0;
      resources->waitFailoverHintEpoch = 0;
      resources->waitFailoverHintAbsStep = 0;
      resources->waitFailoverHintLastSendMs = 0;
      resources->waitFailoverHintSendCount = 0;
      for (int i=0; i<groupSize; i++) sub[-i].groupSize = groupSize;
      ncclProfilerStartRecvProxyOpEvent(s, args);


      static int forceRecvBackupChannels = -1;
      if (forceRecvBackupChannels == -1) {
        const char* env = getenv("NCCL_FORCE_BACKUP_CHANNELS");
        forceRecvBackupChannels = env ? atoi(env) : 0;
      }
      
      if (forceRecvBackupChannels) {
        if(sub->channelId == 0 || sub->channelId == 8){
          resources->useBackup = 1;
        }
      }

      if (sub->reg && sub->nbytes > 0) {
        // Register buffer with both comms for consistency
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvComm, sub->recvbuff, sub->nbytes, NCCL_PTR_CUDA, &sub->mhandle));
        NCCLCHECK(proxyState->ncclNet->regMr(resources->netRecvCommBackup, sub->recvbuff, sub->nbytes, NCCL_PTR_CUDA, &sub->mhandleBackup));
        INFO(NCCL_R2CC, "RECV: Channel %d registered memory with both PRIMARY and BACKUP comms, buffer=%p, size=%ld", 
             sub->channelId, sub->recvbuff, sub->nbytes);
      } else {
        // For pre-registered buffers, copy both handles from resources  
        sub->mhandle = resources->mhandles[args->protocol];
        sub->mhandleBackup = resources->mhandlesBackup[args->protocol];
      }
      r2ccTraceProxyState("RECV", args->id, sub->channelId, R2CC_PROXY_STAGE_READY, sub,
                          resources->useBackup, 0, "op_ready_init", true);
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    // Poll OOB and ingest pending failover requests (apply happens at post/test checkpoints).
    OobNet& oob = OobNet::Get();
    oob.PollHotRepair();

    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      bool failoverApplied = false;
      NCCLCHECK(r2ccRecvApplyPendingFailoverReq(args, subGroup, &failoverApplied));
      if (failoverApplied) {
        args->idle = 0;
        continue;
      }
      struct recvNetResources* subGroupRes = (struct recvNetResources*) (subGroup->connection->transportResources);
      if (subGroupRes->waitFailoverReq) {
        NCCLCHECK(r2ccRecvHandleWaitFailoverReq(subGroup, subGroupRes));
        r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_WAIT_RECV_TEST, subGroup,
                            subGroupRes->useBackup, 0, "wait_failover_req_skip_post", false);
        continue;
      }
      int subCount = 0;
      void* ptrs[NCCL_PROXY_MAX_SUBS];
      int sizes[NCCL_PROXY_MAX_SUBS];
      int tags[NCCL_PROXY_MAX_SUBS];
      void* mhandles[NCCL_PROXY_MAX_SUBS];
      for (int i=0; i<subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        if (sub->posted < sub->nsteps) {
          if (sub->posted >= sub->done + maxDepth) { subCount = 0; break; }
          ncclProfilerStartRecvProxyStepEvents(s+i, args, sub->posted, sub->posted+args->sliceSteps);
          struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
          if (sub->reg) maxDepth = 1;
          int stepSize = resources->buffSizes[p] / NCCL_STEPS;
          char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
          int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
          volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
          if (p == NCCL_PROTO_SIMPLE && resources->shared) {
            if (sub->reg) {
              // Wait until CUDA kernel has started before we access the user buffer directly.
              if (connFifo[sub->base%NCCL_STEPS].size == -1) continue;
              ptrs[subCount] = sub->recvbuff;
              sizes[subCount] = std::min(MAX_NET_SIZE, sub->nbytes);
            } else {
              int sharedBuffSlot = sub->posted%maxDepth;
              int offset;
              NCCLCHECK(sharedBuffersGet(proxyState, sub->channelId, sharedBuffSlot*args->nsubs+s+i, &offset, sizes+subCount));
              connFifo[buffSlot].offset = offset;
              ptrs[subCount] = localBuff+offset;
            }
          } else {
            ptrs[subCount] = localBuff+buffSlot*stepSize;
            sizes[subCount] = stepSize*args->sliceSteps;
          }
          if (sub->nbytes < sizes[subCount]) sizes[subCount] = sub->nbytes;
          tags[subCount] = resources->tpRemoteRank;
          mhandles[subCount] = resources->useBackup ? sub->mhandleBackup : sub->mhandle;
          subCount++;
        }
      }
      if (subCount) {
        struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
        uint64_t step = subGroup->posted;
        //TRACE(NCCL_NET, "[%s] id=%d channel_id=%d, step=%d, [prepare recv], tpRank=%d tpLocalRank=%d tpRemoteRank=%d tpRemoteProxyRank=%d 2. irecv from network", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), 
        //args->id, subGroup->channelId, int(step % NCCL_STEPS), resources->tpRank, resources->tpLocalRank, resources->tpRemoteRank, resources->tpRemoteProxyRank);
        
        
        void** requestPtr = subGroup->requests+(step%NCCL_STEPS);

        // if(subGroup->channelId%2==0)
        //     NCCLCHECK(proxyState->ncclNet->irecv(resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, requestPtr));
        //   else


        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // Log which comm is being used for irecv (only when MODE1 subsystem is enabled)
        if (resources->useBackup) {
          INFO(NCCL_MODE1, "RECV: Channel %d using BACKUP comm for irecv", subGroup->channelId);
        }
        INFO(NCCL_R2CC, "RECV: Calling irecv for channel=%d, step=%ld, useBackup=%d", subGroup->channelId, step, resources->useBackup);
        NCCLCHECK(proxyState->ncclNet->irecv(resources->useBackup ? resources->netRecvCommBackup : resources->netRecvComm, subCount, ptrs, sizes, tags, mhandles, requestPtr));

        if (*requestPtr) {
          INFO(NCCL_R2CC, "RECV: irecv allocated request %p for channel=%d, step=%ld", *requestPtr, subGroup->channelId, step);
          subGroup->recvRequestsCache[step%NCCL_STEPS] = *requestPtr;
          proxyState->ncclNet->setRequestChannel(requestPtr, subGroup->channelId);
          proxyState->ncclNet->setRequestId(requestPtr, args->id);
          proxyState->ncclNet->setRequestComm(requestPtr, resources->useBackup ? (void*)(resources->netRecvCommBackup) : (void*)(resources->netRecvComm));
          proxyState->ncclNet->setRequestStep(requestPtr, step);
          proxyState->ncclNet->setRequestOperation(requestPtr, 1);
          r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_IRECV_POSTED, subGroup,
                              resources->useBackup, 0, "irecv_request_allocated", false);

          subGroup->recvRequestsSubCount = subCount;
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld, useBackup=%d, comm=%p, rank=%d, remoteRank=%d: allocate request success", args->id, subGroup->channelId, step, resources->useBackup,  resources->useBackup ? (void*)(resources->netRecvCommBackup) : (void*)(resources->netRecvComm), resources->tpRank, resources->tpRemoteRank); 
           
          // TRACE(NCCL_NET, "[%s] id=%d channel_id=%d, step=%d [allocate recv request] tpRank=%d tpLocalRank=%d tpRemoteRank=%d tpRemoteProxyRank=%d 2. irecv from network", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), 
          // args->id, subGroup->channelId, int(step % NCCL_STEPS), resources->tpRank, resources->tpLocalRank, resources->tpRemoteRank, resources->tpRemoteProxyRank);
       
                  
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup+i;
            sub->posted += args->sliceSteps;
            ncclProfilerRecordProxyOpEventState(s+i, args, sub->posted, sub->transSize, ncclProfilerProxyOpRecvPosted);
            ncclProfilerRecordProxyStepEventStates(s+i, args, sub->posted-args->sliceSteps, sub->posted, ncclProfilerProxyStepRecvWait);
          }
          args->idle = 0;
        } else {
          r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_WAIT_RECV_TEST, subGroup,
                              resources->useBackup, 0, "irecv_request_not_allocated", false);
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      bool failoverApplied = false;
      NCCLCHECK(r2ccRecvApplyPendingFailoverReq(args, subGroup, &failoverApplied));
      if (failoverApplied) {
        args->idle = 0;
        continue;
      }
      struct recvNetResources* subGroupRes = (struct recvNetResources*) (subGroup->connection->transportResources);
      if (subGroupRes->waitFailoverReq) {
        NCCLCHECK(r2ccRecvHandleWaitFailoverReq(subGroup, subGroupRes));
        r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_WAIT_RECV_TEST, subGroup,
                            subGroupRes->useBackup, 0, "wait_failover_req_skip_test", false);
        continue;
      }
      if (subGroup->posted > subGroup->received) {
        // TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d 3. Test if the receive has completed.", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
        uint64_t step = subGroup->received;
        int done;
        void* ptrs[NCCL_PROXY_MAX_SUBS];
        int sizes[NCCL_PROXY_MAX_SUBS];
        void* mhandles[NCCL_PROXY_MAX_SUBS];
        for (int i=0; i<NCCL_PROXY_MAX_SUBS; i++) sizes[i] = 0;
        struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        done = 0;
        r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_WAIT_RECV_TEST, subGroup,
                            resources->useBackup, 0, "poll_recv_request", false);
        void* reqPtr = subGroup->requests[step%NCCL_STEPS];
        if (reqPtr == NULL) {
          r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_WAIT_RECV_TEST, subGroup,
                              resources->useBackup, 0, "skip_test_null_request", false);
          continue;
        }
        INFO(NCCL_R2CC, "RECV: Testing request %p for channel=%d, step=%ld", reqPtr, subGroup->channelId, step);
        NCCLCHECK(proxyState->ncclNet->test(reqPtr, &done, sizes));
        INFO(NCCL_R2CC, "RECV: Test result done=%d for request %p, channel=%d", done, reqPtr, subGroup->channelId);
        
        //if(done == 0 && !resources->useBackup)
        //  NCCLCHECK(proxyState->ncclNet->testBackup(resources->netRecvCommBackup, &done));
        

        if (done == 0) {
          // No local failure yet. Continue polling until the request completes
          // or the socket transport reports a repairable failure.
          r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_WAIT_RECV_TEST, subGroup,
                              resources->useBackup, 0, "recv_wait_no_local_failover", false);
        }
        else if (done == -1){
          // The receiver observes the failed TCP request first and advertises
          // its safe absolute step. The sender converts that hint into a
          // FAILOVER_REQ, switches to backup, and replays from the hinted step.
          INFO(NCCL_R2CC, "RECV: test returned -1 while waiting sender failover req channel=%d useBackup=%d",
               subGroup->channelId, resources->useBackup);
          uint64_t recvAbsStep = subGroup->base + step;
          uint64_t hintEpoch = resources->lastFailoverEpoch + 1;
          if (hintEpoch == 0) hintEpoch = 1;
          subGroup->requests[step%NCCL_STEPS] = NULL;
          subGroup->recvRequestsCache[step%NCCL_STEPS] = NULL;
          subGroup->recvRequestsSubCount = 0;
          resources->waitFailoverReq = 1;
          resources->waitFailoverStartMs = r2ccNowMs();
          resources->waitFailoverLastWarnMs = 0;
          resources->waitFailoverHintEpoch = hintEpoch;
          resources->waitFailoverHintAbsStep = recvAbsStep;
          resources->waitFailoverHintLastSendMs = 0;
          resources->waitFailoverHintSendCount = 0;
          NCCLCHECK(r2ccRecvSendFailoverHint(subGroup, resources, hintEpoch, recvAbsStep, false));
          WARN("R2CC_RECV failover trigger: test=-1, enter wait_failover_req ch=%d conn=%d epoch=%" PRIu64
               " peer=%d step=%" PRIu64 " absStep=%" PRIu64 " useBackup=%d",
               subGroup->channelId, resources->connIndex, resources->lastFailoverEpoch,
               resources->tpRemoteRank, step, recvAbsStep, resources->useBackup);
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld, useBackup=%d, comm=%p, rank=%d, remoteRank=%d: recv test done=-1 waiting sender OOB",
                args->id, subGroup->channelId, step, resources->useBackup,
                resources->useBackup ? (void*)(resources->netRecvCommBackup) : (void*)(resources->netRecvComm),
                resources->tpRank, resources->tpRemoteRank);
          r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_WAIT_RECV_TEST, subGroup,
                              resources->useBackup, 0, "recv_test_minus1_wait_sender_req", true);
        }
        //TRACE(NCCL_NET, "recvProxyProgress: done=%d, useBackup=%d, channel_id=%d ", done, resources->useBackup, subGroup->channelId);
        // Does the size need to be changed?
        else if (done == 1) { // work done
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld, useBackup=%d, comm=%p, rank=%d, remoteRank=%d:  done=1", args->id, subGroup->channelId, step, resources->useBackup,  resources->useBackup ? (void*)(resources->netRecvCommBackup) : (void*)(resources->netRecvComm), resources->tpRank, resources->tpRemoteRank); 
          r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_RECV_DONE, subGroup,
                              resources->useBackup, 0, "recv_test_done", false);
        
          // TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d done=%d, useBackup=%d, channel_id=%d tpRank=%d tpLocalRank=%d tpRemoteRank=%d tpRemoteProxyRank=%d 2. irecv from network", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id, done, resources->useBackup, subGroup->channelId, resources->tpRank, resources->tpLocalRank, resources->tpRemoteRank, resources->tpRemoteProxyRank);
        
         // TRACE(NCCL_INIT, "test done = 1 , channelId %d, netDev=%d, netDevBackup=%d, useBackup=%d", resources->channelId, resources->netDev, resources->netDevBackup, resources->useBackup);
         // TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d done=%d, useBackup=%d, channel_id=%d tpRank=%d tpLocalRank=%d tpRemoteRank=%d tpRemoteProxyRank=%d 2. irecv from network", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id, done, resources->useBackup, subGroup->channelId, resources->tpRank, resources->tpLocalRank, resources->tpRemoteRank, resources->tpRemoteProxyRank);
        
          //TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d 4. Flush", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
          int needFlush = 0;
          int totalSize = 0;
          int subIndex = 0;
          for (int i=0; i<NCCL_PROXY_MAX_SUBS; i++) totalSize += sizes[i];
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;
            if (sub->received < sub->nsteps) {
              int size = sizes[subIndex++];
              if (sub->reg) {
                if (size < sub->nbytes) {
                  sub->recvbuff += size;
                  sub->nbytes -= size;
                  // Do one more step (at least)
                  sub->nsteps++;
                } else {
                  // Reset connFifo size indicating the GPU was ready to receive.
                  // There is a __sync_synchronize() later to ensure it is reset before it is set again by the GPU.
                  struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
                  volatile struct ncclConnFifo* connFifo = (volatile struct ncclConnFifo*)resources->recvMem->connFifo;
                  connFifo[sub->base%NCCL_STEPS].size = -1;
                }
              }
            }
            sub->received += args->sliceSteps;
            sub->transSize += sizes[i];
            ncclProfilerRecordProxyOpEventState(s+i, args, sub->received, sub->transSize, ncclProfilerProxyOpRecvReceived);
            ncclProfilerRecordProxyStepEventStates(s+i, args, sub->received-args->sliceSteps, sub->received, ncclProfilerProxyStepRecvFlushWait);
            if (step < sub->nsteps) {
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              if (resources->useGdr) needFlush |= resources->needFlush;
            }
          }
          subGroup->requests[step%NCCL_STEPS] = NULL;
          if (totalSize > 0 && p == NCCL_PROTO_SIMPLE && needFlush) {
            // GDRCOPY support
            struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
            if (resources->gdcFlush) {
#if defined (__x86_64__)
              // Force a PCI-E read from GPU memory
              asm volatile ("mov (%0), %%eax" :: "l"(resources->gdcFlush) : "%eax");
#else
              WARN("NET: GDR Flush only supported on x86_64");
              return ncclInternalError;
#endif
            } else {
              int subCount = 0;
              for (int i=0; i<subGroup->groupSize; i++) {
                struct ncclProxySubArgs* sub = subGroup + i;
                if (step < sub->nsteps) {
                  struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
                  int stepSize = resources->buffSizes[p] / NCCL_STEPS;
                  char* localBuff = NCCL_NET_MAP_GET_POINTER(&resources->map, cpu, buffs[p]);
                  int buffSlot = (sub->base+sub->received-args->sliceSteps)%NCCL_STEPS;
                  ptrs[subCount] = resources->shared ?
                    (sub->reg ? (char*)sub->recvbuff : localBuff+resources->recvMem->connFifo[buffSlot].offset) :
                    localBuff+buffSlot*stepSize;
                  mhandles[subCount] = resources->useBackup ? sub->mhandleBackup : sub->mhandle;
                  subCount++;
                }
              }
              struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
              // if(subGroup->channelId%2==0)
              //     NCCLCHECK(proxyState->ncclNet->iflush(resources->netRecvComm, subCount, ptrs, sizes, mhandles, subGroup->requests+(step%NCCL_STEPS)));
              //   else
              NCCLCHECK(proxyState->ncclNet->iflush(resources->useBackup ? resources->netRecvCommBackup : resources->netRecvComm, subCount, ptrs, sizes, mhandles, subGroup->requests+(step%NCCL_STEPS)));
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      if (subGroup->received > subGroup->transmitted) {
        //TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d 5. Test if the flush has completed", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
        uint64_t step = subGroup->transmitted;
        int done = 1;
        void* request = subGroup->requests[step%NCCL_STEPS];
        struct recvNetResources* resources = (struct recvNetResources*) (subGroup->connection->transportResources);
        r2ccTraceProxyState("RECV", args->id, subGroup->channelId, R2CC_PROXY_STAGE_WAIT_FLUSH_TEST, subGroup,
                            resources->useBackup, 0, "poll_flush_request", false);
        if (request) NCCLCHECK(proxyState->ncclNet->test(request, &done, NULL));
        if (done) {
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup + i;

            sub->transmitted += args->sliceSteps;
            ncclProfilerRecordProxyOpEventState(s+i, args, sub->transmitted, sub->transSize, ncclProfilerProxyOpRecvTransmitted);
            ncclProfilerRecordProxyStepEventStates(s+i, args, sub->transmitted-args->sliceSteps, sub->transmitted, ncclProfilerProxyStepRecvGPUWait);
            if (step < sub->nsteps) {
              __sync_synchronize();
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              volatile uint64_t* recvTail = resources->gdcSync ? resources->gdcSync : &resources->recvMem->tail;
              if (sub->reg) {
                // We may have added more net steps, but reg operations only have a single step w.r.t. the GPU.
                if (sub->transmitted == sub->nsteps) *recvTail = sub->base + args->sliceSteps;
              } else
                *recvTail = sub->base + sub->transmitted;
              if (resources->gdcSync) wc_store_fence(); // Flush out WC write
            }
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
      for (int i=0; i<subGroup->groupSize; i++) {
        struct ncclProxySubArgs* sub = subGroup + i;
        if (sub->done == sub->nsteps) continue;
        if (sub->transmitted > sub->done) {
          struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
          volatile uint64_t* sendHead = &resources->sendMem->head;
          uint64_t done = sub->reg ? sub->base + sub->nsteps : *sendHead;
          r2ccTraceProxyState("RECV", args->id, sub->channelId, R2CC_PROXY_STAGE_WAIT_SENDHEAD_ACK, sub,
                              resources->useBackup, 0, "poll_send_head_ack", false);
          while (done > sub->base + sub->done &&
              // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
              sub->transmitted > sub->done) {
            if (subGroup->recvRequestsCache[sub->done%NCCL_STEPS]) {
              // the multirecv requests are only cached in the first sub.
              if (proxyState->ncclNet->irecvConsumed){
                TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d 6. irecvConsumed", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
        
                // if(subGroup->channelId%2==0)
                //   NCCLCHECK(proxyState->ncclNet->irecvConsumed(resources->netRecvComm, subGroup->recvRequestsSubCount, subGroup->recvRequestsCache[sub->done%NCCL_STEPS]));
                // else
                NCCLCHECK(proxyState->ncclNet->irecvConsumed(resources->useBackup ? resources->netRecvCommBackup : resources->netRecvComm, subGroup->recvRequestsSubCount, subGroup->recvRequestsCache[sub->done%NCCL_STEPS]));
                // NCCLCHECK(proxyState->ncclNet->irecvConsumed(resources->netRecvComm, subGroup->recvRequestsSubCount, subGroup->recvRequestsCache[sub->done%NCCL_STEPS]));
              }
              subGroup->recvRequestsCache[sub->done%NCCL_STEPS] = NULL;
            }
            sub->done += args->sliceSteps;
            ncclProfilerStopProxyStepEvents(s+i, args, sub->done-args->sliceSteps, sub->done);
            ncclProfilerRecordProxyOpEventState(s+i, args, sub->done, sub->transSize, ncclProfilerProxyOpRecvDone);
            args->idle = 0;
            if (sub->done == sub->nsteps) {
              struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
              if (sub->reg && sub->nbytes > 0) {
                // Deregister from both comms
                NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvComm, sub->mhandle));
                NCCLCHECK(proxyState->ncclNet->deregMr(resources->netRecvCommBackup, sub->mhandleBackup));
                INFO(NCCL_R2CC, "RECV: Channel %d deregistered memory from both PRIMARY and BACKUP comms", sub->channelId);
              }
              r2ccTraceProxyState("RECV", args->id, sub->channelId, R2CC_PROXY_STAGE_SUB_DONE, sub,
                                  resources->useBackup, 0, "sub_all_steps_done", true);
              args->done++;
              break;
            }
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      //TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d args done", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
      
      struct recvNetResources* resources = (struct recvNetResources*) ((args->subs+0)->connection->transportResources);
      struct ncclProxySubArgs* sub0 = args->subs + 0;
      TRACE(NCCL_NET, "id=%d, comm=%p, rank=%d, remoteRank=%d, args done", args->id, resources->useBackup? resources->netRecvCommBackup : resources->netRecvComm, resources->tpRank, resources->tpRemoteRank); 
      r2ccTraceProxyState("RECV", args->id, sub0->channelId, R2CC_PROXY_STAGE_OP_DONE, sub0,
                          resources->useBackup, 0, "proxy_op_done", true);
      args->state = ncclProxyOpNone;
      for (int s=0; s<args->nsubs; s++) {
        ncclProfilerStopProxyOpEvent(s, args);
      }
    }
  }
  return ncclSuccess;
}

struct ncclTransport netTransport = {
  "NET",
  canConnect,
  { sendSetup, sendConnect, sendFree, proxySharedInit, sendProxySetup, sendProxyConnect, sendProxyFree, sendProxyProgress, NULL },
  { recvSetup, recvConnect, recvFree, proxySharedInit, recvProxySetup, recvProxyConnect, recvProxyFree, recvProxyProgress, NULL }
};
