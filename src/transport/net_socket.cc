/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "param.h"

#include <pthread.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>
#include <fcntl.h>
#include <time.h>

/* Init functions */
static int ncclNetIfs = -1;
struct ncclNetSocketDev {
  union ncclSocketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char* pciPath;
};
static struct ncclNetSocketDev ncclNetSocketDevs[MAX_IFS];

pthread_mutex_t ncclNetSocketLock = PTHREAD_MUTEX_INITIALIZER;

static ncclResult_t ncclNetSocketGetPciPath(char* devName, char** pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, NULL);
  return ncclSuccess;
}

ncclResult_t ncclNetSocketInit(ncclDebugLogger_t logFunction) {
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclNetSocketLock);
    if (ncclNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE*MAX_IFS];
      union ncclSocketAddress addrs[MAX_IFS];
      ncclNetIfs = ncclFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        #define MAX_LINE_LEN (2047)
        char line[MAX_LINE_LEN+1];
        char addrline[SOCKET_NAME_MAXLEN+1];
        line[0] = '\0';
        addrline[SOCKET_NAME_MAXLEN] = '\0';
        for (int i=0; i<ncclNetIfs; i++) {
          strcpy(ncclNetSocketDevs[i].devName, names+i*MAX_IF_NAME_SIZE);
          memcpy(&ncclNetSocketDevs[i].addr, addrs+i, sizeof(union ncclSocketAddress));
          NCCLCHECK(ncclNetSocketGetPciPath(ncclNetSocketDevs[i].devName, &ncclNetSocketDevs[i].pciPath));
          snprintf(line+strlen(line), MAX_LINE_LEN-strlen(line), " [%d]%s:%s", i, names+i*MAX_IF_NAME_SIZE,
              ncclSocketToString(&addrs[i], addrline));
        }
        line[MAX_LINE_LEN] = '\0';
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclNetSocketLock);
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketDevices(int* ndev) {
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

static ncclResult_t ncclNetSocketGetSpeed(char* devName, int* speed) {
  ncclResult_t ret = ncclSuccess;
  *speed = 0;
  char speedPath[PATH_MAX];
  sprintf(speedPath, "/sys/class/net/%s/speed", devName);
  int fd = -1;
  SYSCHECKSYNC(open(speedPath, O_RDONLY), "open", fd);
  if (fd != -1) {
    char speedStr[] = "        ";
    int n;
    // Allow this to silently fail
    n = read(fd, speedStr, sizeof(speedStr)-1);
    if (n > 0) {
      *speed = strtol(speedStr, NULL, 0);
    }
  }
  if (*speed <= 0) {
    INFO(NCCL_NET, "Could not get speed from %s. Defaulting to 10 Gbps.", speedPath);
    *speed = 10000;
  }
  if (fd != -1) SYSCHECK(close(fd), "close");
  return ret;
}

ncclResult_t ncclNetSocketGetProperties(int dev, ncclNetProperties_t* props) {
  props->name = ncclNetSocketDevs[dev].devName;
  props->pciPath = ncclNetSocketDevs[dev].pciPath;
  props->guid = dev;
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 0;
  NCCLCHECK(ncclNetSocketGetSpeed(props->name, &props->speed));
  props->latency = 0; // Not set
  props->port = 0;
  props->maxComms = 65536;
  props->maxRecvs = 1;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS NCCL_NET_MAX_REQUESTS
#define MIN_CHUNKSIZE (64*1024)

NCCL_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
NCCL_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);
NCCL_PARAM(SocketStallTimeout, "SOCKET_STALL_TIMEOUT", 5);
NCCL_PARAM(SocketPeerAck, "SOCKET_PEER_ACK", 1);

static inline uint64_t ncclNetSocketNowNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000 * 1000 * 1000 + uint64_t(ts.tv_nsec);
}

enum ncclNetSocketCommState {
  ncclNetSocketCommStateStart = 0,
  ncclNetSocketCommStateConnect = 1,
  ncclNetSocketCommStateAccept = 3,
  ncclNetSocketCommStateSend = 4,
  ncclNetSocketCommStateRecv = 5,
};

enum ncclNetSocketRequestState {
  ncclNetSocketRequestStateFree = 0,
  ncclNetSocketRequestStateSize = 1,
  ncclNetSocketRequestStatePayload = 2,
  ncclNetSocketRequestStateAck = 3,
  ncclNetSocketRequestStateDone = 4,
  ncclNetSocketRequestStateFailed = 5,
};

struct ncclNetSocketCommStage {
  enum ncclNetSocketCommState state;
  uint8_t iteration;
  struct ncclSocket* sock;
  struct ncclNetSocketComm* comm;
};

struct ncclNetSocketHandle {
  union ncclSocketAddress connectAddr;
  uint64_t magic; // random number to help debugging
  int nSocks;
  int nThreads;
  struct ncclNetSocketCommStage stage;
};

struct ncclNetSocketTask {
  int op;
  void* data;
  int size;
  struct ncclSocket* sock;
  int offset;
  int used;
  ncclResult_t result;
  uint64_t lastProgressNs;
};

struct ncclNetSocketAck {
  uint64_t seq;
  int size;
};

struct ncclNetSocketRequest {
  int op;
  void* data;
  int size;
  int ctrlData;
  int ctrlOffset;
  struct ncclSocket* ctrlSock;
  struct ncclNetSocketAck ackData;
  int ackOffset;
  struct ncclSocket* ackSock;
  int offset;
  int used;
  uint64_t seq;
  struct ncclNetSocketComm* comm;
  struct ncclNetSocketTask* tasks[MAX_SOCKETS];
  int nSubs;
  uint64_t lastProgressNs;
  // R2CC fields for backup support
  int channel;
  int id;
  void* netComm;
  int step;
  int operation;  // 1=recv, 2=send
};

struct ncclNetSocketTaskQueue {
  int next;
  int len;
  struct ncclNetSocketTask* tasks;
};

struct ncclNetSocketThreadResources {
  struct ncclNetSocketTaskQueue threadTaskQueue;
  int stop;
  struct ncclNetSocketComm* comm;
  pthread_mutex_t threadLock;
  pthread_cond_t  threadCond;
};

struct ncclNetSocketListenComm {
  struct ncclSocket sock;
  struct ncclNetSocketCommStage stage;
  int nSocks;
  int nThreads;
  int dev;
};

struct ncclNetSocketComm {
  struct ncclSocket ctrlSock;
  struct ncclSocket ackSock;
  struct ncclSocket socks[MAX_SOCKETS];
  int dev;
  int cudaDev;
  int nSocks;
  int nThreads;
  int nextSock;
  uint64_t nextRequestSeq;
  int failed;
  struct ncclNetSocketRequest requests[MAX_REQUESTS];
  pthread_t helperThread[MAX_THREADS];
  struct ncclNetSocketThreadResources threadResources[MAX_THREADS];
};

static inline bool ncclNetSocketProgressTimedOut(uint64_t lastProgressNs, uint64_t nowNs) {
  int64_t timeoutSec = ncclParamSocketStallTimeout();
  uint64_t timeoutNs = uint64_t(timeoutSec) * 1000 * 1000 * 1000;
  return timeoutNs != 0 && lastProgressNs != 0 && nowNs > lastProgressNs && (nowNs - lastProgressNs) >= timeoutNs;
}

static inline void ncclNetSocketMarkFailed(struct ncclNetSocketComm* comm) {
  if (comm) __atomic_store_n(&comm->failed, 1, __ATOMIC_RELAXED);
}

static inline bool ncclNetSocketUpdateProgress(uint64_t* lastProgressNs, int prevOffset, int offset) {
  if (offset <= prevOffset) return false;
  *lastProgressNs = ncclNetSocketNowNs();
  return true;
}

static inline bool ncclNetSocketRequestActive(struct ncclNetSocketRequest* r) {
  return r->used != ncclNetSocketRequestStateFree;
}

static inline void ncclNetSocketReleaseTasks(struct ncclNetSocketRequest* r) {
  for (int i=0; i<r->nSubs; i++) {
    struct ncclNetSocketTask* sub = r->tasks[i];
    if (sub) {
      sub->used = 0;
      r->tasks[i] = NULL;
    }
  }
  r->nSubs = 0;
}

static inline void ncclNetSocketRequestDone(struct ncclNetSocketRequest* r) {
  ncclNetSocketReleaseTasks(r);
  r->lastProgressNs = ncclNetSocketNowNs();
  r->used = ncclNetSocketRequestStateDone;
}

static inline void ncclNetSocketReapRequest(struct ncclNetSocketRequest* r, int* done, int* size) {
  if (size) *size = r->size;
  *done = 1;
  ncclNetSocketReleaseTasks(r);
  r->used = ncclNetSocketRequestStateFree;
}

static inline void ncclNetSocketStartAck(struct ncclNetSocketRequest* r) {
  memset(&r->ackData, 0, sizeof(r->ackData));
  if (r->op == NCCL_SOCKET_RECV) {
    r->ackData.seq = r->seq;
    r->ackData.size = r->size;
  }
  r->ackOffset = 0;
  r->lastProgressNs = ncclNetSocketNowNs();
  r->used = ncclNetSocketRequestStateAck;
}

void* persistentSocketThread(void *args_) {
  struct ncclNetSocketThreadResources* resource = (struct ncclNetSocketThreadResources*)args_;
  struct ncclNetSocketComm* comm = resource->comm;
  struct ncclNetSocketTaskQueue* myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  while (1) {
    int idle = 1;
    int mark = myQueue->next; // mark newest task seen
    for (int i=0; i<myQueue->len; i+=nSocksPerThread) {
      int repeat;
      do {
        repeat = 0;
        for (int j=0; j<nSocksPerThread; j++) {
          struct ncclNetSocketTask* r = myQueue->tasks+i+j;
          if (r != NULL && r->used == 1 && r->offset < r->size) {
            int prevOffset = r->offset;
            r->result = ncclSocketProgress(r->op, r->sock, r->data, r->size, &r->offset);
            if (r->result != ncclSuccess) {
              ncclNetSocketMarkFailed(comm);
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            if (r->offset > prevOffset) {
              __atomic_store_n(&r->lastProgressNs, ncclNetSocketNowNs(), __ATOMIC_RELAXED);
            }
            idle = 0;
            if (r->offset < r->size) repeat = 1;
          }
        }
      } while (repeat);
    }
    if (idle) {
      pthread_mutex_lock(&resource->threadLock);
      while (mark == myQueue->next && resource->stop == 0) { // no new tasks, wait
        pthread_cond_wait(&resource->threadCond, &resource->threadLock);
      }
      pthread_mutex_unlock(&resource->threadLock);
    }
    if (resource->stop) return NULL;
  }
}

ncclResult_t ncclNetSocketGetNsockNthread(int dev, int* ns, int* nt) {
  ncclResult_t ret = ncclSuccess;
  int nSocksPerThread = ncclParamSocketNsocksPerThread();
  int nThreads = ncclParamSocketNthreads();
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : NCCL_SOCKET_NTHREADS is greater than the maximum allowed, setting to %d", MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  int fd = -1;
  int nSocks;
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt=0, autoNs=1; // By default, we only use the main thread and do not spawn extra threads
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor", ncclNetSocketDevs[dev].devName);
    // Coverity is wrong.  NULL second argument to realpath() is OK by POSIX.1-2008.
    // coverity[alias_transfer:FALSE]
    char* rPath = realpath(vendorPath, NULL);
    fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(NCCL_NET, "Open of %s failed : %s", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    SYSCHECKGOTO(read(fd, vendor, 6), "read", ret, fail);
    if (strcmp(vendor, "0x1d0f") == 0) { // AWS
      autoNt = 2;
      autoNs = 8;
    } else if (strcmp(vendor, "0x1ae0") == 0) { // GCP
      autoNt = 4;
      autoNs = 1;
    }
end:
    if (nThreads == -2) nThreads = autoNt;
    if (nSocksPerThread == -2) nSocksPerThread = autoNs;
  }
  nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS/nThreads;
    WARN("NET/Socket : the total number of sockets is greater than the maximum allowed, setting NCCL_NSOCKS_PERTHREAD to %d", nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  if (nSocks > 0) INFO(NCCL_INIT, "NET/Socket: Using %d threads and %d sockets per thread", nThreads, nSocksPerThread);
exit:
  if (fd != -1) close(fd);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclNetSocketListen(int dev, void* opaqueHandle, void** listenComm) {
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }
  ncclResult_t ret = ncclSuccess;
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  memset(handle, 0, sizeof(struct ncclNetSocketHandle));
  static_assert(sizeof(struct ncclNetSocketHandle) <= NCCL_NET_HANDLE_MAXSIZE, "ncclNetSocketHandle size too large");
  struct ncclNetSocketListenComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));
  handle->magic = NCCL_SOCKET_MAGIC;
  NCCLCHECKGOTO(ncclSocketInit(&comm->sock, &ncclNetSocketDevs[dev].addr, handle->magic, ncclSocketTypeNetSocket, NULL, 1), ret, fail);
  NCCLCHECKGOTO(ncclSocketListen(&comm->sock), ret, fail);
  NCCLCHECKGOTO(ncclSocketGetAddr(&comm->sock, &handle->connectAddr), ret, fail);
  NCCLCHECKGOTO(ncclNetSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads), ret, fail);
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  comm->dev = dev;
  *listenComm = comm;
exit:
  return ret;
fail:
  (void)ncclSocketClose(&comm->sock);
  free(comm);
  goto exit;
}

ncclResult_t ncclNetSocketConnect(int dev, void* opaqueHandle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  if (dev < 0 || dev >= ncclNetIfs) { // data transfer socket is based on specified dev
    return ncclInternalError;
  }

  int ready;
  struct ncclNetSocketHandle* handle = (struct ncclNetSocketHandle*) opaqueHandle;
  struct ncclNetSocketCommStage* stage = &handle->stage;
  struct ncclNetSocketComm* comm = stage->comm;
  uint8_t i = stage->iteration;
  struct ncclSocket* sock = stage->sock;
  *sendComm = NULL;

  if (stage->state == ncclNetSocketCommStateConnect) goto socket_connect_check;
  if (stage->state == ncclNetSocketCommStateSend) goto socket_send;

  NCCLCHECK(ncclCalloc(&comm, 1));
  stage->comm = comm;
  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;
  comm->dev = dev;
  CUDACHECK(cudaGetDevice(&comm->cudaDev));
  for (; i<comm->nSocks+2; i++) {
    sock = (i == comm->nSocks) ? &comm->ctrlSock :
      (i == comm->nSocks+1) ? &comm->ackSock : comm->socks+i;
    NCCLCHECK(ncclSocketInit(sock, &handle->connectAddr, handle->magic, ncclSocketTypeNetSocket, NULL, 1));

    stage->sock = sock;
    stage->state = ncclNetSocketCommStateConnect;
    stage->iteration = i;
    NCCLCHECK(ncclSocketConnect(sock));

socket_connect_check:
    NCCLCHECK(ncclSocketReady(sock, &ready));
    if (! ready) return ncclSuccess;
    stage->state = ncclNetSocketCommStateSend;

socket_send:
    int done = 0;
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_SEND, sock, &i, sizeof(uint8_t), &done));
    if (done == 0) return ncclSuccess;
  }
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  struct ncclNetSocketListenComm* lComm = (struct ncclNetSocketListenComm*)listenComm;
  struct ncclNetSocketCommStage* stage = &lComm->stage;
  struct ncclNetSocketComm* rComm = stage->comm;
  uint8_t i = stage->iteration;
  struct ncclSocket* sock = stage->sock;
  int ready;

  *recvComm = NULL;
  if (stage->state == ncclNetSocketCommStateAccept) goto socket_accept_check;
  if (stage->state == ncclNetSocketCommStateRecv) goto socket_recv;

  NCCLCHECK(ncclCalloc(&rComm, 1));
  stage->comm = rComm;
  rComm->nSocks = lComm->nSocks;
  rComm->nThreads = lComm->nThreads;
  rComm->dev = lComm->dev;
  CUDACHECK(cudaGetDevice(&rComm->cudaDev));
  for (; i<rComm->nSocks+2; i++) {
    uint8_t sendSockIdx;

    NCCLCHECK(ncclCalloc(&sock, 1));
    NCCLCHECK(ncclSocketInit(sock));
    stage->sock = sock;
    stage->state = ncclNetSocketCommStateAccept;
    stage->iteration = i;
    NCCLCHECK(ncclSocketAccept(sock, &lComm->sock));

socket_accept_check:
    NCCLCHECK(ncclSocketReady(sock, &ready));
    if (!ready) return ncclSuccess;

    stage->state = ncclNetSocketCommStateRecv;
socket_recv:
    int done = 0;
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, sock, &sendSockIdx, sizeof(uint8_t), &done));
    if (done == 0) return ncclSuccess;

    if (sendSockIdx == rComm->nSocks)
      memcpy(&rComm->ctrlSock, sock, sizeof(struct ncclSocket));
    else if (sendSockIdx == rComm->nSocks+1)
      memcpy(&rComm->ackSock, sock, sizeof(struct ncclSocket));
    else
      memcpy(rComm->socks+sendSockIdx, sock, sizeof(struct ncclSocket));
    free(sock);
  }
  *recvComm = rComm;

  /* reset lComm state */
  stage->state = ncclNetSocketCommStateStart;
  stage->iteration = 0;
  stage->sock = NULL;
  stage->comm = NULL;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketGetRequest(struct ncclNetSocketComm* comm, int op, void* data, int size, struct ncclNetSocketRequest** req) {
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclNetSocketRequest* r = comm->requests+i;
    if (r->used == ncclNetSocketRequestStateFree) {
      uint64_t nowNs = ncclNetSocketNowNs();
      r->op = op;
      r->data = data;
      r->size = size;
      r->ctrlData = size;
      r->ctrlOffset = 0;
      r->ctrlSock = &comm->ctrlSock;
      memset(&r->ackData, 0, sizeof(r->ackData));
      r->ackOffset = 0;
      r->ackSock = &comm->ackSock;
      r->offset = 0;
      r->used = ncclNetSocketRequestStateSize;
      r->seq = comm->nextRequestSeq++;
      r->comm = comm;
      r->nSubs = 0;
      r->lastProgressNs = nowNs;
      r->channel = -1;
      r->id = -1;
      r->netComm = NULL;
      r->step = -1;
      r->operation = 0;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("NET/Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclNetSocketGetTask(struct ncclNetSocketComm* comm, int op, void* data, int size, struct ncclNetSocketTask** req) {
  int tid = comm->nextSock % comm->nThreads;
  struct ncclNetSocketThreadResources* res = comm->threadResources+tid;
  struct ncclNetSocketTaskQueue* queue = &res->threadTaskQueue;
  // create helper threads and prepare per-thread task queue
  if (queue->tasks == NULL) {
    // each request can be divided up to nSocks tasks, and
    // these tasks are distributed to nThreads threads,
    // we need to make sure each thread queue has enough slots for MAX_REQUESTS
    queue->len = MAX_REQUESTS * DIVUP(comm->nSocks, comm->nThreads);
    NCCLCHECK(ncclCalloc(&queue->tasks, queue->len));
    queue->next = 0;
    res->comm = comm;
    pthread_mutex_init(&res->threadLock, NULL);
    pthread_cond_init(&res->threadCond, NULL);
    PTHREADCHECK(pthread_create(comm->helperThread+tid, NULL, persistentSocketThread, res), "pthread_create");
    ncclSetThreadName(comm->helperThread[tid], "NCCL Sock%c%1u%2u%2u", op == NCCL_SOCKET_SEND ? 'S' : 'R', comm->dev, tid, comm->cudaDev);
  }
  struct ncclNetSocketTask* r = queue->tasks+queue->next;
  if (r->used == 0) {
    r->op = op;
    r->data = data;
    r->size = size;
    r->sock = comm->socks + comm->nextSock;
    r->offset = 0;
    r->result = ncclSuccess;
    r->lastProgressNs = ncclNetSocketNowNs();
    comm->nextSock = (comm->nextSock + 1) % comm->nSocks;
    r->used = 1;
    *req = r;
    pthread_mutex_lock(&res->threadLock);
    queue->next = (queue->next+1)%queue->len;
    pthread_cond_signal(&res->threadCond);
    pthread_mutex_unlock(&res->threadLock);
    return ncclSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return ncclInternalError;
}

static ncclResult_t ncclNetSocketProgressSize(struct ncclNetSocketRequest* r, int* ctrlBlocked) {
  int prevOffset = r->ctrlOffset;
  ncclResult_t ret = ncclSocketProgress(r->op, r->ctrlSock, &r->ctrlData, sizeof(int), &r->ctrlOffset);

  if (ret != ncclSuccess) {
    r->used = ncclNetSocketRequestStateFailed;
    ncclNetSocketMarkFailed(r->comm);
    WARN("NET/Socket : ctrl socket failure");
    return ncclSuccess;
  }

  if (!ncclNetSocketUpdateProgress(&r->lastProgressNs, prevOffset, r->ctrlOffset) &&
      r->ctrlOffset < (int)sizeof(int) &&
      ncclNetSocketProgressTimedOut(r->lastProgressNs, ncclNetSocketNowNs())) {
    r->used = ncclNetSocketRequestStateFailed;
    ncclNetSocketMarkFailed(r->comm);
    WARN("NET/Socket : request stalled");
    return ncclSuccess;
  }

  if (r->ctrlOffset < (int)sizeof(int)) {
    *ctrlBlocked = 1;
    return ncclSuccess;
  }

  // Check size is less or equal to the size provided by the user
  if (r->op == NCCL_SOCKET_RECV && r->ctrlData > r->size) {
    char line[SOCKET_NAME_MAXLEN+1];
    union ncclSocketAddress addr;
    NCCLCHECK(ncclSocketGetAddr(r->ctrlSock, &addr));
    WARN("NET/Socket : peer %s message truncated : receiving %d bytes instead of %d. If you believe your socket network is in healthy state, \
        there may be a mismatch in collective sizes or environment settings (e.g. NCCL_PROTO, NCCL_ALGO) between ranks",
        ncclSocketToString(&addr, line), r->ctrlData, r->size);
    return ncclInvalidUsage;
  }
  r->size = r->ctrlData;
  r->offset = 0;
  r->used = ncclNetSocketRequestStatePayload;

  int chunkOffset = 0, i = 0;
  if (r->comm->nSocks > 0) {
    // Each request can be divided up to nSocks tasks.
    int taskSize = std::max(MIN_CHUNKSIZE, DIVUP(r->size, r->comm->nSocks));
    while (chunkOffset < r->size) {
      int chunkSize = std::min(taskSize, r->size-chunkOffset);
      NCCLCHECK(ncclNetSocketGetTask(r->comm, r->op, (char*)(r->data)+chunkOffset, chunkSize, r->tasks+i++));
      chunkOffset += chunkSize;
    }
  }
  r->nSubs = i;
  return ncclSuccess;
}

static ncclResult_t ncclNetSocketProgressPayload(struct ncclNetSocketRequest* r, int* ctrlBlocked) {
  int payloadDone = 0;
  if (r->nSubs > 0) {
    int nCompleted = 0;
    uint64_t nowNs = ncclNetSocketNowNs();
    for (int i=0; i<r->nSubs; i++) {
      struct ncclNetSocketTask* sub = r->tasks[i];
      if (sub->result != ncclSuccess) {
        r->used = ncclNetSocketRequestStateFailed;
        ncclNetSocketMarkFailed(r->comm);
        WARN("NET/Socket : subtask failure");
        return ncclSuccess;
      }
      if (sub->offset == sub->size) {
        nCompleted++;
      } else if (ncclNetSocketProgressTimedOut(__atomic_load_n(&sub->lastProgressNs, __ATOMIC_RELAXED), nowNs)) {
        r->used = ncclNetSocketRequestStateFailed;
        ncclNetSocketMarkFailed(r->comm);
        WARN("NET/Socket : request stalled");
        return ncclSuccess;
      }
    }
    if (nCompleted == r->nSubs) payloadDone = 1;
  } else {
    // With no helper sockets, payload shares ctrlSock and must remain ordered with size headers.
    if (r->offset < r->size) {
      int prevOffset = r->offset;
      ncclResult_t ret = ncclSocketProgress(r->op, r->ctrlSock, r->data, r->size, &r->offset);
      if (ret != ncclSuccess) {
        r->used = ncclNetSocketRequestStateFailed;
        ncclNetSocketMarkFailed(r->comm);
        WARN("NET/Socket : ctrl socket failure");
        return ncclSuccess;
      }
      if (!ncclNetSocketUpdateProgress(&r->lastProgressNs, prevOffset, r->offset) &&
          r->offset < r->size &&
          ncclNetSocketProgressTimedOut(r->lastProgressNs, ncclNetSocketNowNs())) {
        r->used = ncclNetSocketRequestStateFailed;
        ncclNetSocketMarkFailed(r->comm);
        WARN("NET/Socket : request stalled");
        return ncclSuccess;
      }
    }
    if (r->offset == r->size) {
      payloadDone = 1;
    } else {
      *ctrlBlocked = 1;
    }
  }

  if (payloadDone) {
    ncclNetSocketReleaseTasks(r);
    if (ncclParamSocketPeerAck()) {
      // For TCP, local send completion only means the kernel accepted bytes.
      // Wait for a peer ACK so blackholed traffic is visible to the stall timer.
      ncclNetSocketStartAck(r);
    } else {
      ncclNetSocketRequestDone(r);
    }
  }
  return ncclSuccess;
}

static ncclResult_t ncclNetSocketProgressAck(struct ncclNetSocketRequest* r, int* ackBlocked) {
  int ackOp = r->op == NCCL_SOCKET_SEND ? NCCL_SOCKET_RECV : NCCL_SOCKET_SEND;
  int prevOffset = r->ackOffset;
  ncclResult_t ret = ncclSocketProgress(ackOp, r->ackSock, &r->ackData, sizeof(r->ackData), &r->ackOffset);
  if (ret != ncclSuccess) {
    r->used = ncclNetSocketRequestStateFailed;
    ncclNetSocketMarkFailed(r->comm);
    WARN("NET/Socket : peer ACK socket failure");
    return ncclSuccess;
  }
  if (!ncclNetSocketUpdateProgress(&r->lastProgressNs, prevOffset, r->ackOffset) &&
      r->ackOffset < (int)sizeof(r->ackData) &&
      ncclNetSocketProgressTimedOut(r->lastProgressNs, ncclNetSocketNowNs())) {
    r->used = ncclNetSocketRequestStateFailed;
    ncclNetSocketMarkFailed(r->comm);
    WARN("NET/Socket : peer ACK stalled");
    return ncclSuccess;
  }
  if (r->ackOffset < (int)sizeof(r->ackData)) {
    *ackBlocked = 1;
    return ncclSuccess;
  }
  if (r->op == NCCL_SOCKET_SEND && (r->ackData.seq != r->seq || r->ackData.size != r->size)) {
    r->used = ncclNetSocketRequestStateFailed;
    ncclNetSocketMarkFailed(r->comm);
    WARN("NET/Socket : peer ACK mismatch : received seq %llu size %d expected seq %llu size %d",
         (unsigned long long)r->ackData.seq, r->ackData.size, (unsigned long long)r->seq, r->size);
    return ncclSuccess;
  }
  ncclNetSocketRequestDone(r);
  return ncclSuccess;
}

static ncclResult_t ncclNetSocketProgressComm(struct ncclNetSocketComm* comm) {
  if (comm == NULL) return ncclInternalError;
  if (__atomic_load_n(&comm->failed, __ATOMIC_RELAXED)) return ncclSuccess;

  int ctrlBlocked = 0;
  int ackBlocked = 0;
  uint64_t lastSeq = 0;
  int haveLastSeq = 0;

  while (1) {
    struct ncclNetSocketRequest* r = NULL;
    for (int i=0; i<MAX_REQUESTS; i++) {
      struct ncclNetSocketRequest* candidate = comm->requests+i;
      if (!ncclNetSocketRequestActive(candidate)) continue;
      if (haveLastSeq && candidate->seq <= lastSeq) continue;
      if (r == NULL || candidate->seq < r->seq) r = candidate;
    }
    if (r == NULL) break;
    lastSeq = r->seq;
    haveLastSeq = 1;

    if (r->used == ncclNetSocketRequestStateFailed) {
      ncclNetSocketMarkFailed(comm);
      return ncclSuccess;
    }

    if (r->used == ncclNetSocketRequestStateSize) {
      if (!ctrlBlocked) NCCLCHECK(ncclNetSocketProgressSize(r, &ctrlBlocked));
    }

    if (r->used == ncclNetSocketRequestStatePayload) {
      NCCLCHECK(ncclNetSocketProgressPayload(r, &ctrlBlocked));
    }

    if (ackBlocked) continue;
    if (r->used == ncclNetSocketRequestStateAck) {
      NCCLCHECK(ncclNetSocketProgressAck(r, &ackBlocked));
    } else if (ncclParamSocketPeerAck() && r->used != ncclNetSocketRequestStateDone) {
      // Keep ACK stream ordered. Later requests may move payload, but their ACKs wait here.
      ackBlocked = 1;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclNetSocketRequest *r = (struct ncclNetSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }

  NCCLCHECK(ncclNetSocketProgressComm(r->comm));

  if (r->comm && __atomic_load_n(&r->comm->failed, __ATOMIC_RELAXED)) {
    *done = -1;
    return ncclSuccess;
  }
  if (r->used == ncclNetSocketRequestStateFailed) {
    *done = -1;
    return ncclSuccess;
  }
  if (r->used == ncclNetSocketRequestStateDone) {
    ncclNetSocketReapRequest(r, done, size);
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclNetSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclNetSocketIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)sendComm;
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_SEND, data, size, (struct ncclNetSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclNetSocketIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)recvComm;
  if (n != 1) return ncclInternalError;
  NCCLCHECK(ncclNetSocketGetRequest(comm, NCCL_SOCKET_RECV, data[0], sizes[0], (struct ncclNetSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclNetSocketIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclNetSocketCloseListen(void* opaqueComm) {
  struct ncclNetSocketListenComm* comm = (struct ncclNetSocketListenComm*)opaqueComm;
  if (comm) {
    int ready;
    NCCLCHECK(ncclSocketReady(&comm->sock, &ready));
    if (ready) NCCLCHECK(ncclSocketClose(&comm->sock));
    free(comm);
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketClose(void* opaqueComm) {
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)opaqueComm;
  if (comm) {
    for (int i=0; i<comm->nThreads; i++) {
      struct ncclNetSocketThreadResources* res = comm->threadResources+i;
      if (comm->helperThread[i]) {
        pthread_mutex_lock(&res->threadLock);
        res->stop = 1;
        pthread_cond_signal(&res->threadCond);
        pthread_mutex_unlock(&res->threadLock);
        PTHREADCHECK(pthread_join(comm->helperThread[i], NULL), "pthread_join");
      }
      free(res->threadTaskQueue.tasks);
    }
    int ready;
    NCCLCHECK(ncclSocketReady(&comm->ctrlSock, &ready));
    if (ready) NCCLCHECK(ncclSocketClose(&comm->ctrlSock));
    NCCLCHECK(ncclSocketReady(&comm->ackSock, &ready));
    if (ready) NCCLCHECK(ncclSocketClose(&comm->ackSock));
    for (int i=0; i<comm->nSocks; i++) {
      NCCLCHECK(ncclSocketReady(&comm->socks[i], &ready));
      if (ready) NCCLCHECK(ncclSocketClose(&comm->socks[i]));
    }
    free(comm);
  }
  return ncclSuccess;
}

// R2CC backup support functions for Socket transport
ncclResult_t ncclNetSocketSetBackup(void* sendComm) {
  // Socket doesn't need special backup setup
  return ncclSuccess;
}

ncclResult_t ncclNetSocketTestBackup(void* recvComm, int* done) {
  // For socket, we don't have a separate backup test mechanism
  *done = 0;
  return ncclSuccess;
}

ncclResult_t ncclNetSocketSetRequestChannel(void** request, int channel) {
  if (request && *request) {
    ((struct ncclNetSocketRequest*)(*request))->channel = channel;
  }
  return ncclSuccess;
}

int ncclNetSocketGetRequestChannel(void* request) {
  if (request) {
    return ((struct ncclNetSocketRequest*)(request))->channel;
  }
  return -1;
}

ncclResult_t ncclNetSocketSetRequestId(void** request, int id) {
  if (request && *request) {
    ((struct ncclNetSocketRequest*)(*request))->id = id;
  }
  return ncclSuccess;
}

int ncclNetSocketGetRequestId(void* request) {
  if (request) {
    return ((struct ncclNetSocketRequest*)(request))->id;
  }
  return -1;
}

ncclResult_t ncclNetSocketSetRequestComm(void** request, void* comm) {
  if (request && *request) {
    ((struct ncclNetSocketRequest*)(*request))->netComm = comm;
  }
  return ncclSuccess;
}

void* ncclNetSocketGetRequestComm(void* request) {
  if (request) {
    return ((struct ncclNetSocketRequest*)(request))->netComm;
  }
  return NULL;
}

ncclResult_t ncclNetSocketSetRequestStep(void** request, int step) {
  if (request && *request) {
    ((struct ncclNetSocketRequest*)(*request))->step = step;
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketSetRequestOperation(void** request, int op) {
  if (request && *request) {
    ((struct ncclNetSocketRequest*)(*request))->operation = op;
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketCheckSwitchToBackup(void* sendComm, int* change) {
  // For Socket transport, we don't have automatic failure detection
  // The application layer should handle connection failures
  *change = 0;
  if (sendComm == NULL) {
    return ncclSuccess;
  }
  
  struct ncclNetSocketComm* comm = (struct ncclNetSocketComm*)sendComm;
  // Check if the control socket is still connected
  int ready;
  NCCLCHECK(ncclSocketReady(&comm->ctrlSock, &ready));
  if (!ready) {
    // Socket is closed, should switch to backup
    *change = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclNetSocketTimeoutPost(void* comm, void* mhandle) {
  // Socket doesn't have timeout post mechanism
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclNetSocketInit,
  ncclNetSocketDevices,
  ncclNetSocketGetProperties,
  ncclNetSocketListen,
  ncclNetSocketConnect,
  ncclNetSocketAccept,
  ncclNetSocketRegMr,
  NULL, // No DMA-BUF support
  ncclNetSocketDeregMr,
  ncclNetSocketIsend,
  ncclNetSocketIrecv,
  ncclNetSocketIflush,
  ncclNetSocketTest,
  ncclNetSocketClose,
  ncclNetSocketClose,
  ncclNetSocketCloseListen,
  NULL /* getDeviceMr */,
  NULL /* irecvConsumed */,
  ncclNetSocketSetBackup,
  ncclNetSocketTestBackup,
  ncclNetSocketSetRequestChannel,
  ncclNetSocketGetRequestChannel,
  ncclNetSocketSetRequestId,
  ncclNetSocketGetRequestId,
  ncclNetSocketSetRequestComm,
  ncclNetSocketGetRequestComm,
  ncclNetSocketSetRequestStep,
  ncclNetSocketSetRequestOperation,
  ncclNetSocketCheckSwitchToBackup,
  ncclNetSocketTimeoutPost
};
