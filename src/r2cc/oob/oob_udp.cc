#include "r2cc/oob/oob_udp.h"
#include "bootstrap.h"
#include "socket.h"
#include "debug.h"
#include "utils.h"
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <cstdio>
#include <cstdlib>
#include <inttypes.h>
#include <ifaddrs.h>
#include <netdb.h>

OobNet& OobNet::Get() {
  static OobNet instance;
  return instance;
}

static constexpr const char* kHotRepairMsgPrefix = "R2CC_HR";
static constexpr const char* kHotRepairStepPrefix = "R2CC_HR_STEP";
static constexpr const char* kHotRepairStepReqPrefix = "R2CC_HR_STEP_REQ";

// 1.1 Function: GetLocalIP
static ncclResult_t GetLocalIP(char* ipStr, size_t maxLen) {
  union ncclSocketAddress addr;
  char ifName[MAX_IF_NAME_SIZE];
  
  // Use NCCL's utility to find the best interface (respects NCCL_SOCKET_IFNAME)
  if (ncclFindInterfaces(ifName, &addr, MAX_IF_NAME_SIZE, 1) != 1) {
    WARN("OOB: Unable to find any valid network interface");
    return ncclSystemError;
  }
  
  if (addr.sa.sa_family == AF_INET) {
      inet_ntop(AF_INET, &addr.sin.sin_addr, ipStr, maxLen);
  } else if (addr.sa.sa_family == AF_INET6) {
      inet_ntop(AF_INET6, &addr.sin6.sin6_addr, ipStr, maxLen);
  } else {
      WARN("OOB: Unsupported address family");
      return ncclSystemError;
  }

  INFO(NCCL_INIT, "OOB: using interface %s IP %s", ifName, ipStr);
  return ncclSuccess;
}

// 1.2 Function: InitSocket
ncclResult_t OobNet::InitSocket(const char* ipStr) {
  // 1. Create UDP Socket
  sockfd_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (sockfd_ < 0) {
    WARN("OOB: Failed to create UDP socket");
    return ncclSystemError;
  }

  // 2. Bind to Specific IP found by NCCL
  memset(&myAddr_, 0, sizeof(myAddr_));
  myAddr_.sin_family = AF_INET;
  if (inet_pton(AF_INET, ipStr, &myAddr_.sin_addr) <= 0) {
      WARN("OOB: Invalid IP string for bind");
      return ncclSystemError;
  }
  myAddr_.sin_port = 0; // Let OS choose ephemeral port

  if (bind(sockfd_, (struct sockaddr *)&myAddr_, sizeof(myAddr_)) < 0) {
    WARN("OOB: Failed to bind UDP socket to %s", ipStr);
    return ncclSystemError;
  }

  // 3. Get assigned port
  socklen_t len = sizeof(myAddr_);
  if (getsockname(sockfd_, (struct sockaddr *)&myAddr_, &len) < 0) {
    WARN("OOB: Failed to get socket name");
    return ncclSystemError;
  }
  
  // 4. Set non-blocking
  int flags = fcntl(sockfd_, F_GETFL, 0);
  fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK);

  INFO(NCCL_INIT, "OOB: Socket Initialized on Port %d", ntohs(myAddr_.sin_port));
  return ncclSuccess;
}

// 1.3 Function: ExchangeInfo
ncclResult_t OobNet::ExchangeInfo(void* bootstrapHandle, const char* myIP) {
  PeerOobInfo myInfo;
  myInfo.port = ntohs(myAddr_.sin_port);
  strncpy(myInfo.ip_str, myIP, INET_ADDRSTRLEN);
  
  PeerOobInfo* allInfos = (PeerOobInfo*)malloc(nRanks_ * sizeof(PeerOobInfo));
  if (!allInfos) return ncclSystemError;
  
  // Zero out first to be safe
  memset(allInfos, 0, nRanks_ * sizeof(PeerOobInfo));
  memcpy(&allInfos[rank_], &myInfo, sizeof(PeerOobInfo));
  
  NCCLCHECK(bootstrapAllGather(bootstrapHandle, allInfos, sizeof(PeerOobInfo)));

  // Store peer addresses
  peerStates_.resize(nRanks_);
  for (int i = 0; i < nRanks_; ++i) {
    memset(&peerStates_[i].addr, 0, sizeof(peerStates_[i].addr));
    peerStates_[i].addr.sin_family = AF_INET;
    peerStates_[i].addr.sin_port = htons(allInfos[i].port);
    inet_pton(AF_INET, allInfos[i].ip_str, &peerStates_[i].addr.sin_addr);
    
    INFO(NCCL_INIT, "OOB: Rank %d discovered peer %d at %s:%d", rank_, i, allInfos[i].ip_str, allInfos[i].port);
  }

  free(allInfos);
  return ncclSuccess;
}

// 1.4 Function: TestSend (Verification)
ncclResult_t OobNet::TestSend(int peerRank) {
  if (peerRank < 0 || peerRank >= nRanks_) return ncclInvalidArgument;
  
  const char* msg = "PING";
  sendto(sockfd_, msg, 5, 0, 
         (struct sockaddr*)&peerStates_[peerRank].addr, sizeof(peerStates_[peerRank].addr));
         
  INFO(NCCL_INIT, "OOB: TestSend - Sent PING to Rank %d", peerRank);
  verificationState = "SENT_PING"; // Record state
  return ncclSuccess;
}

// 1.5 Function: TestRecv (Verification)
ncclResult_t OobNet::TestRecv(int expectedCount) {
  char buf[32];
  struct sockaddr_in srcAddr;
  socklen_t addrLen = sizeof(srcAddr);
  
  int receivedCount = 0;
  int retries = 0;
  
  // Wait loop
  while (retries < 3000) { // Wait up to ~3 seconds
      ssize_t n = recvfrom(sockfd_, buf, sizeof(buf), 0, (struct sockaddr*)&srcAddr, &addrLen);
      if (n > 0) {
          receivedCount++;
          INFO(NCCL_INIT, "OOB: TestRecv - Received '%s' from %s:%d (%d/%d)", 
               buf, inet_ntoa(srcAddr.sin_addr), ntohs(srcAddr.sin_port), receivedCount, expectedCount);
          
          if (receivedCount >= expectedCount) {
              verificationState = "RECEIVED_ALL"; // Record state
              return ncclSuccess;
          }
      } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
          WARN("OOB: TestRecv error: %s", strerror(errno));
          verificationState = "RECV_ERROR";
          return ncclSystemError;
      }
      usleep(1000); // 1ms
      retries++;
  }
  
  WARN("OOB: TestRecv Timeout - Received %d/%d messages", receivedCount, expectedCount);
  verificationState = "RECV_TIMEOUT";
  return ncclSystemError;
}

ncclResult_t OobNet::NotifyHotRepairOnce() {
  bool expected = false;
  if (!hotRepairSent_.compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
    return ncclSuccess; // Already sent.
  }

  if (sockfd_ < 0 || rank_ < 0 || nRanks_ <= 0) {
    return ncclInvalidUsage;
  }

  if (nRanks_ == 1) return ncclSuccess;

  char msg[64];
  int n = snprintf(msg, sizeof(msg), "%s %d", kHotRepairMsgPrefix, rank_);
  if (n <= 0) return ncclSystemError;

  for (int peer = 0; peer < nRanks_; ++peer) {
    if (peer == rank_) continue;
    (void)sendto(sockfd_, msg, (size_t)n + 1, 0,
                 (struct sockaddr*)&peerStates_[peer].addr, sizeof(peerStates_[peer].addr));
  }

  INFO(NCCL_R2CC, "OOB: Broadcast HOT_REPAIR signal from rank %d to %d peers", rank_, nRanks_ - 1);
  return ncclSuccess;
}

ncclResult_t OobNet::SendStepSync(int peerRank, int channelId, uint64_t absStep) {
  if (peerRank < 0 || peerRank >= nRanks_) return ncclInvalidArgument;
  if (channelId < 0 || channelId >= kMaxChannels) return ncclInvalidArgument;
  if (sockfd_ < 0 || rank_ < 0 || nRanks_ <= 0) return ncclInvalidUsage;

  char msg[96];
  int n = snprintf(msg, sizeof(msg), "%s %d %" PRIu64, kHotRepairStepPrefix, channelId, absStep);
  if (n <= 0) return ncclSystemError;

  (void)sendto(sockfd_, msg, (size_t)n + 1, 0,
               (struct sockaddr*)&peerStates_[peerRank].addr, sizeof(peerStates_[peerRank].addr));
  INFO(NCCL_R2CC, "OOB: Sent STEP_SYNC channel=%d step=%" PRIu64 " to rank %d",
       channelId, absStep, peerRank);
  return ncclSuccess;
}

ncclResult_t OobNet::SendStepSyncRequest(int peerRank, int channelId) {
  if (peerRank < 0 || peerRank >= nRanks_) return ncclInvalidArgument;
  if (channelId < 0 || channelId >= kMaxChannels) return ncclInvalidArgument;
  if (sockfd_ < 0 || rank_ < 0 || nRanks_ <= 0) return ncclInvalidUsage;

  char msg[96];
  int n = snprintf(msg, sizeof(msg), "%s %d %d", kHotRepairStepReqPrefix, channelId, rank_);
  if (n <= 0) return ncclSystemError;

  (void)sendto(sockfd_, msg, (size_t)n + 1, 0,
               (struct sockaddr*)&peerStates_[peerRank].addr, sizeof(peerStates_[peerRank].addr));
  INFO(NCCL_R2CC, "OOB: Sent STEP_SYNC_REQUEST channel=%d to rank %d", channelId, peerRank);
  return ncclSuccess;
}

void OobNet::UpdatePendingSyncStep(int channelId, uint64_t absStep) {
  if (channelId < 0 || channelId >= kMaxChannels) return;
  uint64_t encoded = absStep + 1; // allow absStep==0
  uint64_t cur = pendingSyncStep_[channelId].load(std::memory_order_relaxed);
  while (encoded > cur &&
         !pendingSyncStep_[channelId].compare_exchange_weak(
             cur, encoded, std::memory_order_relaxed, std::memory_order_relaxed)) {
  }
  if (encoded > cur) {
    pendingSyncMask_.fetch_or(1ull << channelId, std::memory_order_relaxed);
  }
}

bool OobNet::ConsumeStepSync(int channelId, uint64_t* absStep) {
  if (channelId < 0 || channelId >= kMaxChannels) return false;
  uint64_t cur = pendingSyncStep_[channelId].load(std::memory_order_relaxed);
  while (cur != 0) {
    if (pendingSyncStep_[channelId].compare_exchange_weak(
            cur, 0, std::memory_order_relaxed, std::memory_order_relaxed)) {
      uint64_t decoded = cur - 1;
      if (absStep) *absStep = decoded;
      uint64_t bit = 1ull << channelId;
      pendingSyncMask_.fetch_and(~bit, std::memory_order_relaxed);
      // If a newer step arrived concurrently, re-set the bit.
      if (pendingSyncStep_[channelId].load(std::memory_order_relaxed) != 0) {
        pendingSyncMask_.fetch_or(bit, std::memory_order_relaxed);
      }
      return true;
    }
    // cur updated by compare_exchange
  }
  return false;
}

bool OobNet::ConsumeStepSyncRequest(int channelId, int* peerRank) {
  if (channelId < 0 || channelId >= kMaxChannels) return false;
  int cur = pendingSyncReq_[channelId].load(std::memory_order_relaxed);
  while (cur != 0) {
    if (pendingSyncReq_[channelId].compare_exchange_weak(
            cur, 0, std::memory_order_relaxed, std::memory_order_relaxed)) {
      int decoded = cur - 1;
      if (peerRank) *peerRank = decoded;
      return true;
    }
  }
  return false;
}

void OobNet::ReportFailedChannel(int channelId) {
  if (channelId < 0 || channelId >= 64) return;
  uint64_t bit = 1ull << channelId;
  localFailedChannelMask_.fetch_or(bit, std::memory_order_relaxed);
}

bool OobNet::PollHotRepair() {
  if (hotRepairSeen_.load(std::memory_order_relaxed)) return true;
  if (sockfd_ < 0) return false;

  bool got = false;
  std::lock_guard<std::mutex> lock(recvMutex_);
  while (true) {
    char buf[64];
    struct sockaddr_in srcAddr;
    socklen_t addrLen = sizeof(srcAddr);
    ssize_t n = recvfrom(sockfd_, buf, sizeof(buf) - 1, 0, (struct sockaddr*)&srcAddr, &addrLen);
    if (n > 0) {
      buf[n] = '\0';
      if (strncmp(buf, kHotRepairStepReqPrefix, strlen(kHotRepairStepReqPrefix)) == 0) {
        int channelId = -1;
        int peerRank = -1;
        if (sscanf(buf + strlen(kHotRepairStepReqPrefix), "%d %d", &channelId, &peerRank) == 2) {
          if (channelId >= 0 && channelId < kMaxChannels && peerRank >= 0) {
            pendingSyncReq_[channelId].store(peerRank + 1, std::memory_order_relaxed);
            INFO(NCCL_R2CC, "OOB: Received STEP_SYNC_REQUEST channel=%d from rank %d", channelId, peerRank);
          }
        }
        continue;
      }
      if (strncmp(buf, kHotRepairStepPrefix, strlen(kHotRepairStepPrefix)) == 0) {
        // Best-effort parse: "R2CC_HR_STEP <channel> <step>"
        int channelId = -1;
        unsigned long long step = 0;
        if (sscanf(buf + strlen(kHotRepairStepPrefix), "%d %llu", &channelId, &step) == 2) {
          UpdatePendingSyncStep(channelId, (uint64_t)step);
          INFO(NCCL_R2CC, "OOB: Received STEP_SYNC channel=%d step=%llu from %s:%d",
               channelId, step, inet_ntoa(srcAddr.sin_addr), ntohs(srcAddr.sin_port));
        }
        continue;
      }
      if (strncmp(buf, kHotRepairMsgPrefix, strlen(kHotRepairMsgPrefix)) == 0) {
        got = true;
        hotRepairSeen_.store(true, std::memory_order_relaxed);

        // Best-effort parse: "R2CC_HR <rank>"
        int peerRank = -1;
        char* endptr = nullptr;
        long v = strtol(buf + strlen(kHotRepairMsgPrefix), &endptr, 10);
        // If there is a space before the number, strtol will skip it.
        if (endptr != buf + strlen(kHotRepairMsgPrefix)) peerRank = (int)v;
        if (peerRank >= 0) lastHotRepairPeer_.store(peerRank, std::memory_order_relaxed);

        INFO(NCCL_R2CC, "OOB: Received HOT_REPAIR signal '%s' from %s:%d",
             buf, inet_ntoa(srcAddr.sin_addr), ntohs(srcAddr.sin_port));
      }
      continue;
    }

    if (n == -1 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
    if (n == -1) WARN("OOB: PollHotRepair recvfrom error: %s", strerror(errno));
    break;
  }

  return got || hotRepairSeen_.load(std::memory_order_relaxed);
}

ncclResult_t OobNet::Init(int rank, int nRanks, void* bootstrapHandle) {
  rank_ = rank;
  nRanks_ = nRanks;
  hotRepairSeen_.store(false, std::memory_order_relaxed);
  hotRepairSent_.store(false, std::memory_order_relaxed);
  lastHotRepairPeer_.store(-1, std::memory_order_relaxed);
  for (int i = 0; i < kMaxChannels; ++i) {
    pendingSyncReq_[i].store(0, std::memory_order_relaxed);
  }
  localFailedChannelMask_.store(0, std::memory_order_relaxed);
  globalFailedChannelMask_.store(0, std::memory_order_relaxed);
  pendingSyncMask_.store(0, std::memory_order_relaxed);
  for (int i = 0; i < kMaxChannels; i++) pendingSyncStep_[i].store(0, std::memory_order_relaxed);
  verificationState = "INIT_START";
  
  char myIP[INET_ADDRSTRLEN];
  NCCLCHECK(GetLocalIP(myIP, INET_ADDRSTRLEN));
  
  NCCLCHECK(InitSocket(myIP));
  NCCLCHECK(ExchangeInfo(bootstrapHandle, myIP));
  
  // R2CC_PHASE1_VERIFICATION: All-to-One (Everyone sends to Rank 0)
  NCCLCHECK(bootstrapBarrier(bootstrapHandle, rank, nRanks, 0x42));

  if (nRanks > 1) {
    if (rank == 0) {
        // Rank 0 Expects Ping from everyone else
        NCCLCHECK(TestRecv(nRanks - 1));
    } else {
        // Everyone else Pings Rank 0
        NCCLCHECK(TestSend(0));
    }
  } else {
      verificationState = "SINGLE_RANK";
  }

  // Final Barrier to ensure verification is done before moving on
  NCCLCHECK(bootstrapBarrier(bootstrapHandle, rank, nRanks, 0x43));
  
  return ncclSuccess;
}

// Implement the C-export helper
extern "C" void r2cc_oob_get_test_result(char* buf, int maxlen) {
    std::string s = OobNet::Get().verificationState;
    strncpy(buf, s.c_str(), maxlen);
    // Ensure null termination safely
    if (maxlen > 0) buf[maxlen - 1] = '\0';
}
