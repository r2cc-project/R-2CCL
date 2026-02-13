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
NCCL_PARAM(R2CCFailoverTimeoutMs, "R2CC_FAILOVER_TIMEOUT_MS", -1);


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

int send_total_count = 0;
int log_counter = 0;
int isend_counter=0;
int if_posted_counter=0;
int if_transmitted_counter=0;
int if_reg_counter=0;

static ncclResult_t sendProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
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
    }
    struct ncclProxySubArgs* sub = args->subs+0;
    struct sendNetResources* resources =  (struct sendNetResources*) (sub->connection->transportResources);
    TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: init ncclProxyOpReady", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    // Process any pending OOB step-sync messages and align sender progress before continuing.
    OobNet& oob = OobNet::Get();
    oob.PollHotRepair();
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      uint64_t syncAbs = 0;
      if (oob.ConsumeStepSync(sub->channelId, &syncAbs)) {
        struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
        resources->stepSyncRequested = 0;
        if (syncAbs >= sub->base && syncAbs <= sub->base + sub->nsteps) {
          uint64_t syncRel = syncAbs - sub->base;
          // Ignore if receiver claims progress beyond what we've posted.
          if (syncRel <= sub->posted) {
            uint64_t oldDone = sub->done;
            // Clear in-flight requests >= syncRel to avoid stale completions.
            for (uint64_t i = syncRel; i < sub->transmitted; i++) {
              int buffSlot = (sub->base + i) % NCCL_STEPS;
              sub->requests[buffSlot] = NULL;
            }
            sub->transmitted = syncRel;
            sub->done = syncRel;
            sub->mhandle = resources->mhandlesBackup[args->protocol];
            resources->useBackup = 1;
            if (resources->shared == 0 && syncRel > oldDone) {
              volatile uint64_t* sendHead = resources->gdcSync ? resources->gdcSync : &resources->sendMem->head;
              if (sub->reg) {
                // reg operations only have one net step w.r.t. GPU
                *sendHead = sub->base + args->sliceSteps;
              } else {
                *sendHead = sub->base + sub->done;
              }
              if (resources->gdcSync) wc_store_fence();
            }
            INFO(NCCL_R2CC, "SEND: Step-sync applied channel=%d absStep=%" PRIu64 " (rel=%" PRIu64 "), switch to backup",
                 sub->channelId, syncAbs, syncRel);
          } else {
            INFO(NCCL_R2CC, "SEND: Step-sync ignored (ahead of posted) channel=%d absStep=%" PRIu64,
                 sub->channelId, syncAbs);
          }
        } else {
          INFO(NCCL_R2CC, "SEND: Step-sync ignored (out of range) channel=%d absStep=%" PRIu64
               " base=%" PRIu64 " nsteps=%" PRIu64,
               sub->channelId, syncAbs, sub->base, (uint64_t)sub->nsteps);
        }
      }
    }

    // If any sub is waiting for step sync response, pause progress.
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
      if (resources->stepSyncRequested) {
        args->idle = 1;
        return ncclSuccess;
      }
    }


    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (sub->done == sub->nsteps) continue;
      struct sendNetResources* resources = (struct sendNetResources*) (sub->connection->transportResources);
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


        // transmitted == recvTail && transmitted all allocated successful, but didn't got a 
        if ((sub->reg || connFifo[buffSlot].size != -1) && ((*recvTail > tail) || p == NCCL_PROTO_LL)) {

          if_reg_counter++;
          if(if_reg_counter %1000007==0){
            TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d, if_reg_counter++", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
          }

          // We have something to receive, let's check if it's completely ready.
          int size = sub->reg ? std::min(MAX_NET_SIZE, sub->nbytes) : connFifo[buffSlot].size;
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
              sub->transmitted += args->sliceSteps;
              ncclProfilerRecordProxyOpEventState(s, args, sub->transmitted, sub->transSize, ncclProfilerProxyOpSendTransmitted);
              ncclProfilerRecordProxyStepEventStates(s, args, sub->transmitted-args->sliceSteps, sub->transmitted, ncclProfilerProxyStepSendWait);
              sub->transSize += size;
              args->idle = 0;
              continue;
            }
            else{
              log_counter++;
              if(log_counter %1000007==0){
                TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: allocate request failed", args->id, sub->channelId, sub->base+sub->transmitted, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
              }

              if(!resources->useBackup){
                    int change = 0;
                    proxyState->ncclNet->checkSwitchToBackup(resources->netSendCommBackup, &change);
                    if(change == 1){
                      int done;
                      int size;
                      int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
                      // std::this_thread::sleep_for(std::chrono::milliseconds(10));
                      // NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));
                      TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: got message from receiver, need to switch to backup", args->id, sub->channelId, sub->base+sub->done, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
                      for (int s=0; s<args->nsubs; s++) {
                        struct ncclProxySubArgs* sub = args->subs+s;
                        for(int i = sub->done; i < sub->transmitted; i++){
                          int buffSlot = (sub->base + i)%NCCL_STEPS;
                          sub->requests[buffSlot] = NULL;
                          // ((struct ncclIbRequest*)(sub->requests[buffSlot]))->type = NCCL_NET_IB_REQ_UNUSED;;
                        }
                        sub->transmitted = sub->done;
                        // sub->transSize -= size;
                        // std::min(MAX_NET_SIZE, sub->nbytes) : connFifo[buffSlot].size;
                        sub->mhandle = resources->mhandlesBackup[args->protocol];
                      }
                      // R2CC: First time we switch to the backup path, record failed channel and notify
                      // peers via OOB so they can switch to a balanced schedule (R2CC_MODE=2).
                      OobNet::Get().ReportFailedChannel(sub->channelId);
                      OobNet::Get().NotifyHotRepairOnce();
                      resources->useBackup = 1;
                      return ncclSuccess;
                    } else {
                      // Prevent busy-waiting
                      // std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }
                  }

            }
          }
        }
      }
      // Check whether the network has completed some send operations.
      if (sub->done < sub->transmitted) {
        //TRACE(NCCL_NET, "sendproxyprogress: [%s] id=%d 3. Check whether the network has completed some send operations.", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), args->id);
        int done;
        int size;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;
        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
        INFO(NCCL_R2CC, "SEND: Testing request %p for channel=%d, buffSlot=%d", sub->requests[buffSlot], sub->channelId, buffSlot);
        NCCLCHECK(proxyState->ncclNet->test(sub->requests[buffSlot], &done, &size));
        INFO(NCCL_R2CC, "SEND: Test result done=%d for request %p, channel=%d", done, sub->requests[buffSlot], sub->channelId);
        if (done == -1){
          // std::this_thread::sleep_for(std::chrono::milliseconds(120000));

          // std::this_thread::sleep_for(std::chrono::milliseconds(5000));
          //if(!resources->useBackup)
          //  NCCLCHECK(proxyState->ncclNet->setBackup(resources->netSendCommBackup));

          // return ncclInternalError;
          INFO(NCCL_R2CC, "SEND ERROR PATH: test returned -1, channel=%d, useBackup=%d - THIS SHOULD NOT HAPPEN WITH MODE=1", sub->channelId, resources->useBackup);
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld useBackup=%d, comm=%p, rank=%d, remoteRank=%d: test done=-1", args->id, sub->channelId, sub->base+sub->done, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
          for (int s=0; s<args->nsubs; s++) {
            struct ncclProxySubArgs* sub = args->subs+s;
            for(int i = sub->done; i < sub->transmitted; i++){
              int buffSlot = (sub->base + i)%NCCL_STEPS;
              sub->requests[buffSlot] = NULL;
              // ((struct ncclIbRequest*)(sub->requests[buffSlot]))->type = NCCL_NET_IB_REQ_UNUSED;;
            }
            sub->transmitted = sub->done;
            // sub->transSize -= size;
            // std::min(MAX_NET_SIZE, sub->nbytes) : connFifo[buffSlot].size;
            sub->mhandle = resources->mhandlesBackup[args->protocol];
          }
          if (!resources->useBackup) {
            OobNet::Get().SendStepSyncRequest(resources->tpRemoteRank, sub->channelId);
            resources->stepSyncRequested = 1;
            OobNet::Get().ReportFailedChannel(sub->channelId);
            OobNet::Get().NotifyHotRepairOnce();
          }
          resources->useBackup = 1;
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
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      sendNetResources* resources = (struct sendNetResources*) ((args->subs+0)->connection->transportResources);
      TRACE(NCCL_NET, "id=%d, channel=%d, useBackup=%d, comm=%p, rank=%d, remoteRank=%d: args done", args->id, (args->subs+0)->channelId, resources->useBackup, resources->useBackup ? resources->netSendCommBackup : resources->netSendComm, resources->tpRank, resources->tpRemoteRank);
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

static inline void r2ccRecvRollbackAllSubs(struct ncclProxyArgs* args) {
  for (int s = 0; s < args->nsubs; ++s) {
    struct ncclProxySubArgs* sub = args->subs + s;
    for (int i = 0; i < NCCL_STEPS; ++i) {
      sub->requests[i] = NULL;
      sub->timeStamp[i] = -1;
      sub->recvRequestsCache[i] = NULL;
    }
    sub->recvRequestsSubCount = 0;
    sub->posted = sub->received;
  }
}

static ncclResult_t recvProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  
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
      for(int i = 0; i < NCCL_STEPS; i++){
        sub->timeStamp[i] = -1;
      }
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
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    // Process any pending OOB step-sync from sender and align receiver state.
    OobNet& oob = OobNet::Get();
    oob.PollHotRepair();
    bool syncApplied = false;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      uint64_t syncAbs = 0;
      if (oob.ConsumeStepSync(sub->channelId, &syncAbs)) {
        struct recvNetResources* resources = (struct recvNetResources*) (sub->connection->transportResources);
        if (syncAbs >= sub->base && syncAbs <= sub->base + sub->nsteps) {
          uint64_t syncRel = syncAbs - sub->base;
          if (syncRel <= sub->posted) {
            for (uint64_t i = syncRel; i < sub->posted; ++i) {
              int buffSlot = i % NCCL_STEPS;
              sub->requests[buffSlot] = NULL;
              sub->timeStamp[buffSlot] = -1;
              sub->recvRequestsCache[buffSlot] = NULL;
            }
            sub->posted = syncRel;
          }
          if (sub->received > syncRel) sub->received = syncRel;
          if (sub->transmitted > syncRel) sub->transmitted = syncRel;
          if (sub->done > syncRel) sub->done = syncRel;
          resources->useBackup = 1;
          syncApplied = true;
          INFO(NCCL_R2CC, "RECV: Step-sync applied channel=%d absStep=%" PRIu64 " (rel=%" PRIu64 "), switch to backup",
               sub->channelId, syncAbs, syncRel);
        } else {
          INFO(NCCL_R2CC, "RECV: Step-sync ignored (out of range) channel=%d absStep=%" PRIu64
               " base=%" PRIu64 " nsteps=%" PRIu64,
               sub->channelId, syncAbs, sub->base, (uint64_t)sub->nsteps);
        }
      }
    }
    // If any step-sync was applied, force ungrouping to avoid comm mixing.
    if (syncApplied) r2ccForceUngroup(args);

    int p = args->protocol;
    int maxDepth = std::min(NCCL_STEPS, NCCL_SHARED_STEPS/args->nsubs);
    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
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

          subGroup->recvRequestsSubCount = subCount;
          subGroup->timeStamp[step % NCCL_STEPS] = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch()).count();
           
          
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld, useBackup=%d, comm=%p, rank=%d, remoteRank=%d: allocate request success", args->id, subGroup->channelId, step, resources->useBackup,  resources->useBackup ? (void*)(resources->netRecvCommBackup) : (void*)(resources->netRecvComm), resources->tpRank, resources->tpRemoteRank); 
           
          // TRACE(NCCL_NET, "[%s] id=%d channel_id=%d, step=%d, timeStamp: %ld, [allocate recv request] tpRank=%d tpLocalRank=%d tpRemoteRank=%d tpRemoteProxyRank=%d 2. irecv from network", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), 
          // args->id, subGroup->channelId, int(step % NCCL_STEPS), subGroup->timeStamp[step % NCCL_STEPS], resources->tpRank, resources->tpLocalRank, resources->tpRemoteRank, resources->tpRemoteProxyRank);
       
                  
          for (int i=0; i<subGroup->groupSize; i++) {
            struct ncclProxySubArgs* sub = subGroup+i;
            sub->posted += args->sliceSteps;
            ncclProfilerRecordProxyOpEventState(s+i, args, sub->posted, sub->transSize, ncclProfilerProxyOpRecvPosted);
            ncclProfilerRecordProxyStepEventStates(s+i, args, sub->posted-args->sliceSteps, sub->posted, ncclProfilerProxyStepRecvWait);
          }
          args->idle = 0;
        }
      }
    }
    if (args->idle == 0) return ncclSuccess;

    for (int s=0; s<args->nsubs; s+=args->subs[s].groupSize) {
      struct ncclProxySubArgs* subGroup = args->subs+s;
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
        INFO(NCCL_R2CC, "RECV: Testing request %p for channel=%d, step=%ld", subGroup->requests[step%NCCL_STEPS], subGroup->channelId, step);
        NCCLCHECK(proxyState->ncclNet->test(subGroup->requests[step%NCCL_STEPS], &done, sizes));
        INFO(NCCL_R2CC, "RECV: Test result done=%d for request %p, channel=%d", done, subGroup->requests[step%NCCL_STEPS], subGroup->channelId);
        
        //if(done == 0 && !resources->useBackup)
        //  NCCLCHECK(proxyState->ncclNet->testBackup(resources->netRecvCommBackup, &done));
        

        if (done == 2){ // update timer.
          subGroup->timeStamp[step % NCCL_STEPS] = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count();
        }
        else if (done == 0) { // check timeout.
          auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count();
          int64_t timeoutMs = ncclParamR2CCFailoverTimeoutMs();
          if (timeoutMs <= 0) {
            int64_t timeout = 1;
            for (int i = 0; i < ncclParamRecvTimeout() - 20; i++) {
              timeout *= 2;
            }
            timeout *= 4 * (ncclParamRecvRetryCnt() + 1);
            timeoutMs = timeout * 1000;
          }
          // if(subGroup->channelId%2==1)continue;
          if (now - subGroup->timeStamp[step % NCCL_STEPS] > timeoutMs) {
            TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld, useBackup=%d, comm=%p, rank=%d, remoteRank=%d: set timeout request", args->id, subGroup->channelId, step, resources->useBackup,  resources->useBackup ? (void*)(resources->netRecvCommBackup) : (void*)(resources->netRecvComm), resources->tpRank, resources->tpRemoteRank); 
            
            NCCLCHECK(proxyState->ncclNet->ncclIbTimeoutPost(resources->useBackup ? resources->netRecvCommBackup : resources->netRecvComm, subGroup->requests[step%NCCL_STEPS]));
            // Trigger step-sync and failover immediately on timeout to avoid in-flight corruption.
            if (!resources->useBackup) {
              uint64_t syncAbs = subGroup->base + subGroup->received;
              OobNet::Get().SendStepSync(resources->tpRemoteRank, subGroup->channelId, syncAbs);
              OobNet::Get().ReportFailedChannel(subGroup->channelId);
              OobNet::Get().NotifyHotRepairOnce();
            }
            // Roll back posted to received and switch to backup.
            r2ccRecvRollbackAllSubs(args);
            resources->useBackup = 1;
            r2ccForceUngroup(args);
            subGroup->timeStamp[step % NCCL_STEPS] = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch()).count();
            //TRACE(NCCL_NET, "current time: %ld", now);
            //TRACE(NCCL_NET, "timeStamp: %ld", subGroup->timeStamp[step % NCCL_STEPS]);
            //TRACE(NCCL_NET, "recvProxyProgress: ncclParamRecvTimeout %ld  ncclParamRecvRetryCnt %ld, timeout %ld TimeStamp Timeout", ncclParamRecvTimeout(), ncclParamRecvRetryCnt(), timeout);
            //TRACE(NCCL_NET, "recvProxyProgress: ncclParamRecvTimeout %ld  ncclParamRecvRetryCnt %ld, timeout %ld TimeStamp Timeout", ncclParamRecvTimeout(), ncclParamRecvRetryCnt(), timeout);
            // TRACE(NCCL_NET, "[%s] test done = -1, id=%d channel_id=%d, step=%d, timeStamp: %ld, [allocate recv request] tpRank=%d tpLocalRank=%d tpRemoteRank=%d tpRemoteProxyRank=%d 2. irecv from network", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), 
            //args->id, subGroup->channelId, int(step % NCCL_STEPS), subGroup->timeStamp[step % NCCL_STEPS], resources->tpRank, resources->tpLocalRank, resources->tpRemoteRank, resources->tpRemoteProxyRank);
            args->idle = 0;
            return ncclSuccess;
          }
        }
        else if (done == -1){ // disconnect
          // return ncclInternalError;
          INFO(NCCL_R2CC, "RECV ERROR PATH: test returned -1, channel=%d, useBackup=%d - THIS SHOULD NOT HAPPEN WITH MODE=1", subGroup->channelId, resources->useBackup);
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld, useBackup=%d, comm=%p, rank=%d, remoteRank=%d: test done = -1", args->id, subGroup->channelId, step, resources->useBackup,  resources->useBackup ? (void*)(resources->netRecvCommBackup) : (void*)(resources->netRecvComm), resources->tpRank, resources->tpRemoteRank); 
        
          // Send step-sync to peer so sender can align to the first missing step.
          uint64_t syncAbs = subGroup->base + subGroup->received;
          OobNet::Get().SendStepSync(resources->tpRemoteRank, subGroup->channelId, syncAbs);

          //TRACE(NCCL_INIT, "test done = -1, channelId %d, netDev=%d, netDevBackup=%d, useBackup=%d", resources->channelId, resources->netDev, resources->netDevBackup, resources->useBackup);
          r2ccRecvRollbackAllSubs(args);
          if (!resources->useBackup) {
            OobNet::Get().ReportFailedChannel(subGroup->channelId);
            OobNet::Get().NotifyHotRepairOnce();
          }
          resources->useBackup = 1;
          r2ccForceUngroup(args);

          

          //TRACE(NCCL_NET, "recvProxyProgress: [%s] id=%d done=%d, useBackup=%d, channel_id=%d tpRank=%d tpLocalRank=%d tpRemoteRank=%d tpRemoteProxyRank=%d 2. irecv from network", ([]() { std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); static char buffer[100]; std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now)); return buffer; })(), 
          //args->id, done, resources->useBackup, subGroup->channelId, resources->tpRank, resources->tpLocalRank, resources->tpRemoteRank, resources->tpRemoteProxyRank);
        
          args->idle = 0;
          return ncclSuccess;
        }
        //TRACE(NCCL_NET, "recvProxyProgress: done=%d, useBackup=%d, channel_id=%d ", done, resources->useBackup, subGroup->channelId);
        // Does the size need to be changed?
        else if (done == 1) { // work done
          TRACE(NCCL_NET, "id=%d, channel=%d, step=%ld, useBackup=%d, comm=%p, rank=%d, remoteRank=%d:  done=1", args->id, subGroup->channelId, step, resources->useBackup,  resources->useBackup ? (void*)(resources->netRecvCommBackup) : (void*)(resources->netRecvComm), resources->tpRank, resources->tpRemoteRank); 
        
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
          subGroup->timeStamp[step%NCCL_STEPS] = -1;
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
      TRACE(NCCL_NET, "id=%d, comm=%p, rank=%d, remoteRank=%d, args done", args->id, resources->useBackup? resources->netRecvCommBackup : resources->netRecvComm, resources->tpRank, resources->tpRemoteRank); 
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
