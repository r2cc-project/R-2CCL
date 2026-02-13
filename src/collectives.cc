/*************************************************************************
 * Copyright (c) 2015-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "collectives.h"
#include "enqueue.h"
#include "debug.h"
#include "bootstrap.h"
#include "r2cc/oob/oob_udp.h"
#include "nccl.h"
#include <atomic>
#include <chrono>
#include <cstdlib>

const char* ncclFuncToString(ncclFunc_t fn) {
  switch (fn) {
  case ncclFuncAllGather: return "AllGather";
  case ncclFuncAllReduce: return "AllReduce";
  case ncclFuncBroadcast: return "Broadcast";
  case ncclFuncRecv: return "Recv";
  case ncclFuncReduce: return "Reduce";
  case ncclFuncReduceScatter: return "ReduceScatter";
  case ncclFuncSendRecv: return "SendRecv";
  case ncclFuncSend: return "Send";
  default: return "Invalid";
  }
}

const char* ncclDevRedOpToString(ncclDevRedOp_t op) {
  switch (op) {
  case ncclDevSum: return "Sum";
  case ncclDevProd: return "Prod";
  case ncclDevMinMax: return "MinMax";
  case ncclDevPreMulSum: return "PreMulSum";
  case ncclDevSumPostDiv: return "SumPostDiv";
  default: return "Unknown";
  }
}

const char* ncclDatatypeToString(ncclDataType_t type) {
  switch (type) {
  case ncclInt8: return "ncclInt8";
  case ncclInt32: return "ncclInt32";
  case ncclUint32: return "ncclUint32";
  case ncclInt64: return "ncclInt64";
  case ncclUint64: return "ncclUint64";
  case ncclFloat16: return "ncclFloat16";
  case ncclFloat32: return "ncclFloat32";
  case ncclFloat64: return "ncclFloat64";
#if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16: return "ncclBfloat16";
#endif
  default: return "Unknown";
  }
}

const char* ncclAlgoToString(int algo) {
  switch (algo) {
  case NCCL_ALGO_TREE: return "TREE";
  case NCCL_ALGO_RING: return "RING";
  case NCCL_ALGO_COLLNET_DIRECT: return "COLLNET_DIRECT";
  case NCCL_ALGO_COLLNET_CHAIN: return "COLLNET_CHAIN";
  case NCCL_ALGO_NVLS: return "NVLS";
  case NCCL_ALGO_NVLS_TREE: return "NVLS_TREE";
  case NCCL_ALGO_PAT: return "PAT";
  default: return "Unknown";
  }
}

const char* ncclProtoToString(int proto) {
  switch (proto) {
  case NCCL_PROTO_LL: return "LL";
  case NCCL_PROTO_LL128: return "LL128";
  case NCCL_PROTO_SIMPLE: return "SIMPLE";
  default: return "Unknown";
  }
}

NCCL_API(ncclResult_t, ncclAllGather, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  // Just pass the size of one message and not the total bytes sent/received.
  constexpr nvtxPayloadSchemaEntry_t AllGatherSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"}
  };
  size_t msgsize = sendcount * ncclTypeSize(datatype);
  NVTX3_FUNC_WITH_PARAMS(AllGather, AllGatherSchema, msgsize)

  struct ncclInfo info = { ncclFuncAllGather, "AllGather",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct NvtxParamsAllReduce {
    size_t bytes;
    ncclRedOp_t op;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsAllReduce, op)}
  };
  NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)


  size_t totalUserBytes = count * ncclTypeSize(datatype);
  bool parallel = totalUserBytes <= 134217728;

  // auto t0 = std::chrono::high_resolution_clock::now();
  
  // R2CC: If any peer reports a hot-repair event through OOB, gather per-rank failed
  // channel masks via bootstrapAllGather, combine (per-node then global OR), and
  // force Balance mode (R2CC_MODE=2) from this AllReduce onwards.
  {
    const char* env = getenv("R2CC_MODE");
    int curMode = env ? atoi(env) : 0;
    OobNet& oob = OobNet::Get();
    bool seen = oob.PollHotRepair();
    uint64_t localMask = oob.LocalFailedChannelMask();
    uint64_t globalMask = oob.GlobalFailedChannelMask();

    // Only perform the bootstrap allgather once (v1). This requires all ranks to
    // observe the hot-repair signal before entering this branch.
    static std::atomic<int> hrSynced{0};
    if (hrSynced.load(std::memory_order_relaxed) == 0 && (seen || localMask != 0)) {
      int nranks = comm->nRanks;
      uint64_t* allMasks = (uint64_t*)calloc((size_t)nranks, sizeof(uint64_t));
      if (allMasks) {
        allMasks[comm->rank] = localMask;
        NCCLCHECK(bootstrapAllGather(comm->bootstrap, allMasks, sizeof(uint64_t)));

        // Combine per-node first, then global OR.
        int nNodes = comm->nNodes;
        uint64_t* nodeMasks = (uint64_t*)calloc((size_t)nNodes, sizeof(uint64_t));
        if (nodeMasks) {
          for (int r = 0; r < nranks; r++) {
            int node = comm->rankToNode[r];
            if (node >= 0 && node < nNodes) nodeMasks[node] |= allMasks[r];
          }
          uint64_t merged = 0;
          for (int n = 0; n < nNodes; n++) merged |= nodeMasks[n];
          free(nodeMasks);
          globalMask = merged;
        } else {
          // Fallback: global OR across all ranks directly.
          uint64_t merged = 0;
          for (int r = 0; r < nranks; r++) merged |= allMasks[r];
          globalMask = merged;
        }

        oob.SetGlobalFailedChannelMask(globalMask);
        if (comm->rank == 0) {
          INFO(NCCL_R2CC, "R2CC: hot-repair gather complete: localMask=0x%lx globalMask=0x%lx",
               (unsigned long)localMask, (unsigned long)globalMask);
        }
        free(allMasks);
      } else {
        WARN("R2CC: hot-repair gather failed to allocate allMasks");
      }
      hrSynced.store(1, std::memory_order_relaxed);
    }

    if (curMode < 2 && (globalMask != 0 || seen || localMask != 0)) {
      setenv("R2CC_MODE", "2", 1);
      static std::atomic<int> switched{0};
      int expected = 0;
      if (switched.compare_exchange_strong(expected, 1, std::memory_order_relaxed)) {
        INFO(NCCL_R2CC, "R2CC: hot-repair detected, forcing R2CC_MODE=2");
      }
    }
  }


  // R2CC_MODE=3/4/5: Split AllReduce implementation  
  const char* r2ccMode = getenv("R2CC_MODE");
  int mode = r2ccMode ? atoi(r2ccMode) : 0;
  if (mode >= 3 && mode <= 5 && totalUserBytes > 16384) {
    // Get number of network devices to simulate failed NIC
    int nNetDevs;
    NCCLCHECK(comm->ncclNet->devices(&nNetDevs));
    

    if (nNetDevs > 1) {
      // Calculate split: main ring handles (n-1)/n of data
      size_t elementSize = ncclTypeSize(datatype);
      size_t originalCount = count;

      const char* failure2SameServer_str = getenv("FAILURE2_SAME_SERVER");
      int failure2SameServer = failure2SameServer_str ? atoi(failure2SameServer_str) : 0;
      int failure_num = failure2SameServer ? 2 : 1;  

      size_t mainCount = originalCount * (nNetDevs - failure_num) / nNetDevs;
      size_t subCount = originalCount - mainCount;
      
      // MODE=3: Both phases
      // MODE=4: Only Phase 1 (AllReduce)
      // MODE=5: Only Phase 2 (Broadcast)
      if(mode == 3 and parallel){
        NCCLCHECK(ncclGroupStart());
        struct ncclInfo info1 = { ncclFuncAllReduce, "AllReduce",
          sendbuff, recvbuff, mainCount, datatype, op, 0, comm, stream, /* Args */
          ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
        NCCLCHECK(ncclEnqueueCheck(&info1));
        // Calculate offset for remaining data
        size_t offset = mainCount * elementSize;
        const void* bcastSendbuff = (const char*)sendbuff + offset;
        void* bcastRecvbuff = (char*)recvbuff + offset;
        
        struct ncclInfo info2 = { ncclFuncBroadcast, "Broadcast",
          bcastSendbuff, bcastRecvbuff, subCount, datatype, ncclSum /*unused*/, 0 /*root: rank 0*/, comm, stream, /* Args */
          BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
        NCCLCHECK(ncclEnqueueCheck(&info2));
        NCCLCHECK(ncclGroupEnd());
      }
      else if(mode == 3 and !parallel){
	      NCCLCHECK(ncclGroupStart());
        struct ncclInfo info1 = { ncclFuncAllReduce, "AllReduce",
          sendbuff, recvbuff, mainCount, datatype, op, 0, comm, stream, /* Args */
          ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
        NCCLCHECK(ncclEnqueueCheck(&info1));
        NCCLCHECK(ncclGroupEnd());


	      NCCLCHECK(ncclGroupStart());
        // Calculate offset for remaining data
        size_t offset = mainCount * elementSize;
        const void* bcastSendbuff = (const char*)sendbuff + offset;
        void* bcastRecvbuff = (char*)recvbuff + offset;

        struct ncclInfo info2 = { ncclFuncBroadcast, "Broadcast",
          bcastSendbuff, bcastRecvbuff, subCount, datatype, ncclSum /*unused*/, 0 /*root: rank 0*/, comm, stream, /* Args */
          BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
        NCCLCHECK(ncclEnqueueCheck(&info2));
        NCCLCHECK(ncclGroupEnd());

      }
      else if(mode == 4){
        // Phase 1: AllReduce with reduced data
        NCCLCHECK(ncclGroupStart());
        struct ncclInfo info1 = { ncclFuncAllReduce, "AllReduce",
          sendbuff, recvbuff, mainCount, datatype, op, 0, comm, stream, /* Args */
          ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
        NCCLCHECK(ncclEnqueueCheck(&info1));
        NCCLCHECK(ncclGroupEnd());
      }else if(mode == 5){
        // Phase 2: Broadcast remaining data from "failed" node
        NCCLCHECK(ncclGroupStart());
        // Calculate offset for remaining data
        size_t offset = mainCount * elementSize;
        const void* bcastSendbuff = (const char*)sendbuff + offset;
        void* bcastRecvbuff = (char*)recvbuff + offset;
        
        struct ncclInfo info2 = { ncclFuncBroadcast, "Broadcast",
          bcastSendbuff, bcastRecvbuff, subCount, datatype, ncclSum /*unused*/, 0 /*root: rank 0*/, comm, stream, /* Args */
          BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
        NCCLCHECK(ncclEnqueueCheck(&info2));
        NCCLCHECK(ncclGroupEnd());
      }
    }
  }else{
    // Normal AllReduce operation
    struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
      sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
      ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
    NCCLCHECK(ncclEnqueueCheck(&info));
  }

  //CUDACHECK(cudaStreamSynchronize(stream));
  // auto t1 = std::chrono::high_resolution_clock::now();
  // double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // if (comm->rank == 0) {
  //  printf("[R2CC] mode=%d bytes=%zu time=%.3f ms\n",
  //         mode, totalUserBytes, ms);
  // }
 


  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclBroadcast, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  struct NvtxParamsBroadcast {
    size_t bytes;
    int root;
  };
  constexpr nvtxPayloadSchemaEntry_t BroadcastSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsBroadcast, root)}
  };
  NvtxParamsBroadcast payload{count * ncclTypeSize(datatype), root};
  NVTX3_FUNC_WITH_PARAMS(Broadcast, BroadcastSchema, payload)

  struct ncclInfo info = { ncclFuncBroadcast, "Broadcast",
    sendbuff, recvbuff, count, datatype, ncclSum, root, comm, stream, /* Args */
    BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}
/* Deprecated original "in place" function, similar to MPI */
NCCL_API(ncclResult_t, ncclBcast, void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root,
    ncclComm_t comm, cudaStream_t stream) {
  NCCLCHECK(ncclBroadcast(buff, buff, count, datatype, root, comm, stream));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream) {
  struct NvtxParamsReduce {
    size_t bytes;
    int root;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Root", nullptr, 0, offsetof(NvtxParamsReduce, root)},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduce, op)}
  };
  NvtxParamsReduce payload{count * ncclTypeSize(datatype), root, op};
  NVTX3_FUNC_WITH_PARAMS(Reduce, ReduceSchema, payload)

  struct ncclInfo info = { ncclFuncReduce, "Reduce",
    sendbuff, recvbuff, count, datatype, op, root, comm, stream, /* Args */
    REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct NvtxParamsReduceScatter {
    size_t bytes;
    ncclRedOp_t op;
  };
  constexpr nvtxPayloadSchemaEntry_t ReduceScatterSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsReduceScatter, op)}
  };
  NvtxParamsReduceScatter payload{recvcount * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(ReduceScatter, ReduceScatterSchema, payload)

  struct ncclInfo info = { ncclFuncReduceScatter, "ReduceScatter",
    sendbuff, recvbuff, recvcount, datatype, op, 0, comm, stream, /* Args */
    REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS };
  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Send, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncSend, "Send",
    NULL, (void*)sendbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
exit:
  NCCLCHECK(ncclGroupEnd());
  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  NvtxParamsSendRecv payload{count * ncclTypeSize(datatype), peer};
  NVTX3_FUNC_WITH_PARAMS(Recv, SendRecvSchema, payload)

  struct ncclInfo info = { ncclFuncRecv, "Recv",
    NULL, recvbuff, count, datatype, ncclSum, peer, comm, stream, /* Args */
    1, 1 };
  ncclResult_t ret;
  NCCLCHECK(ncclGroupStart());
  NCCLCHECKGOTO(ncclEnqueueCheck(&info), ret, exit);
exit:
  NCCLCHECK(ncclGroupEnd());
  return ret;
}
