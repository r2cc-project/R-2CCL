#ifndef OOB_UDP_H_
#define OOB_UDP_H_

#include "nccl.h"
#include <netinet/in.h>
#include <atomic>
#include <vector>
#include <string>
#include <cstring>
#include <mutex>

// C-style export for testing only
// C-style export for testing only
extern "C" __attribute__((visibility("default"))) void r2cc_oob_get_test_result(char* buf, int maxlen);

class OobNet {
public:
  static OobNet& Get();
  
  // Initialize with rank info and exchange UDP ports
  // bootstrapHandle is void* to avoid circular dependency
  ncclResult_t Init(int rank, int nRanks, void* bootstrapHandle);
  
  // Phase 1 Verification: Simple Ping
  // Phase 1 Verification: Simple Ping
  ncclResult_t TestSend(int peerRank);

  ncclResult_t TestRecv(int expectedCount);

  // Hot-repair notification: used at runtime to coordinate a mode switch.
  // NotifyHotRepairOnce() broadcasts a small UDP message to all peers (best-effort).
  // PollHotRepair() non-blocking checks the socket and latches the signal once received.
  ncclResult_t NotifyHotRepairOnce();
  bool PollHotRepair();
  bool HotRepairSeen() const { return hotRepairSeen_.load(std::memory_order_relaxed); }
  int LastHotRepairPeer() const { return lastHotRepairPeer_.load(std::memory_order_relaxed); }

  // Step synchronization (per channel, best-effort, point-to-point).
  // Receiver sends the absolute step it expects next; sender aligns to that step.
  ncclResult_t SendStepSync(int peerRank, int channelId, uint64_t absStep);
  ncclResult_t SendStepSyncRequest(int peerRank, int channelId);
  bool ConsumeStepSyncRequest(int channelId, int* peerRank);
  bool ConsumeStepSync(int channelId, uint64_t* absStep);

  // Record and query failed channels observed locally and globally.
  void ReportFailedChannel(int channelId);
  uint64_t LocalFailedChannelMask() const { return localFailedChannelMask_.load(std::memory_order_relaxed); }
  void SetGlobalFailedChannelMask(uint64_t mask) { globalFailedChannelMask_.store(mask, std::memory_order_relaxed); }
  uint64_t GlobalFailedChannelMask() const { return globalFailedChannelMask_.load(std::memory_order_relaxed); }
  
private:
  OobNet() : rank_(-1), nRanks_(0), sockfd_(-1) {}
  
  ncclResult_t InitSocket(const char* ipStr);
  ncclResult_t ExchangeInfo(void* bootstrapHandle, const char* myIP);

  int rank_;
  int nRanks_;
  int sockfd_;
  struct sockaddr_in myAddr_;
  
  struct PeerOobInfo {
    char ip_str[INET_ADDRSTRLEN];
    uint16_t port;
  };

  struct PeerState {
    struct sockaddr_in addr;
  };
  
  std::vector<PeerState> peerStates_;

  // Runtime hot-repair signal state.
  std::atomic<bool> hotRepairSeen_{false};
  std::atomic<bool> hotRepairSent_{false};
  std::atomic<int> lastHotRepairPeer_{-1};

  // Pending step sync (encoded as absStep+1 to allow step 0).
  static constexpr int kMaxChannels = 64;
  std::atomic<uint64_t> pendingSyncStep_[kMaxChannels];
  std::atomic<uint64_t> pendingSyncMask_{0};
  std::atomic<int> pendingSyncReq_[kMaxChannels];
  std::mutex recvMutex_;

  // Failed channel state (bitset of channelIds).
  std::atomic<uint64_t> localFailedChannelMask_{0};
  std::atomic<uint64_t> globalFailedChannelMask_{0};

  void UpdatePendingSyncStep(int channelId, uint64_t absStep);
  
  // Verification state for unit tests
  std::string verificationState = "UNINITIALIZED";

  friend void r2cc_oob_get_test_result(char* buf, int maxlen);
};



#endif // OOB_UDP_H_
