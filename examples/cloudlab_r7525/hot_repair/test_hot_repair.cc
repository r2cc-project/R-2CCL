#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <chrono>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

// Basic error handling macros
#define MPICHECK(cmd) do { \
  int e = cmd; \
  if (e != MPI_SUCCESS) { \
    printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define CUDACHECK(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t r = cmd; \
  if (r != ncclSuccess) { \
    printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

// Input of rank r at element i in iteration it: a pseudo-random integer below mask+1, stored as float. mask is chosen
// so that the sum over all ranks stays below 2^24: every summation order then gives the exact result, the output can
// be compared exactly, and data that ends up at the wrong position or iteration does not match what is expected there.
__host__ __device__ inline unsigned int input_value(unsigned long long idx, int rank, int iter, unsigned int mask) {
  unsigned long long x = idx + 0x9E3779B97F4A7C15ull * (unsigned long long)(rank * 4096 + iter + 1);
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x ^= x >> 31;
  return (unsigned int)x & mask;
}

__global__ void fill_kernel(float* buf, size_t n, int rank, int iter, unsigned int mask) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[idx] = (float)input_value(idx, rank, iter, mask);
  }
}

// Compares every element with the exact sum over all ranks; NaN and Inf never match.
__global__ void check_kernel(const float* buf, size_t n, int nRanks, int iter, unsigned int mask,
                             unsigned long long* mismatches, unsigned long long* first_bad) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  unsigned int expect = 0;
  for (int r = 0; r < nRanks; ++r) expect += input_value(idx, r, iter, mask);
  float got = buf[idx];
  if (!isfinite(got) || got != (float)expect) {
    atomicAdd(mismatches, 1ull);
    atomicMin(first_bad, (unsigned long long)idx);
  }
}

struct IbDevCounters {
  std::string dev;
  std::string tx_path;
  std::string rx_path;
  bool available = false;
};

static int get_env_int(const char* name, int def_val) {
  const char* v = getenv(name);
  if (!v || v[0] == '\0') return def_val;
  return atoi(v);
}

static bool read_u64_file(const std::string& path, unsigned long long* val) {
  FILE* f = fopen(path.c_str(), "r");
  if (!f) return false;
  unsigned long long tmp = 0;
  int rc = fscanf(f, "%llu", &tmp);
  fclose(f);
  if (rc != 1) return false;
  *val = tmp;
  return true;
}

static std::vector<IbDevCounters> init_ib_counters() {
  std::vector<IbDevCounters> devs;
  const char* names[] = {"mlx5_0", "mlx5_2", "mlx5_3"};
  for (const char* name : names) {
    IbDevCounters c;
    c.dev = name;
    c.tx_path = std::string("/sys/class/infiniband/") + name + "/ports/1/counters/port_xmit_data";
    c.rx_path = std::string("/sys/class/infiniband/") + name + "/ports/1/counters/port_rcv_data";
    unsigned long long tmp = 0;
    c.available = read_u64_file(c.tx_path, &tmp) && read_u64_file(c.rx_path, &tmp);
    devs.push_back(c);
  }
  return devs;
}

int main(int argc, char* argv[]) {
  int myRank = 0, nRanks = 0;

  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

  int localRank = -1;
  {
    char* localRankStr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (localRankStr) {
      localRank = atoi(localRankStr);
    } else {
      localRank = myRank % 2;
    }
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  printf("[Rank %d] Running on %s (Local Rank %d)\n", myRank, hostname, localRank);

  if (localRank < 0) {
    printf("[Rank %d] Could not determine local rank.\n", myRank);
    MPICHECK(MPI_Finalize());
    return 1;
  }

  const size_t bytes = 4ull * 1024 * 1024 * 1024; // 4 GiB
  const int iters = 10;
  const char* disconnect_cmd = "./nic/disconnect_nic1.sh";
  const char* reconnect_cmd = "./nic/connect_nic1.sh";

  const int start_disconnect_delay_ms = get_env_int("R2CC_AR_START_DISCONNECT_DELAY_MS", -1);
  // Checker self-test: R2CC_TEST_CORRUPT=<n>[,nan] overwrites one output element of rank 0 in iteration n (a wrong
  // value, or NaN) before it is checked. The run must then end in TEST FAIL with a non-zero exit code.
  int corrupt_at = -1, corrupt_nan = 0;
  if (const char* c = getenv("R2CC_TEST_CORRUPT")) {
    corrupt_at = atoi(c);
    corrupt_nan = strstr(c, "nan") != nullptr;
  }
  // Inputs are integers below 2^(24 - rank_bits); the sum over nRanks <= 2^rank_bits ranks stays below 2^24.
  int rank_bits = 0;
  while ((1 << rank_bits) < nRanks) ++rank_bits;
  const unsigned int mask = (1u << (24 - rank_bits)) - 1;

  if (myRank == 0) {
    printf("Config: iters=%d, bytes=%zu (%.2f GiB), count=%zu floats\n",
           iters, bytes, (double)bytes / (1024.0 * 1024.0 * 1024.0), bytes / sizeof(float));
    printf("Config: start_disconnect_delay_ms=%d\n", start_disconnect_delay_ms);
    printf("Config: disconnect_cmd=%s, reconnect_cmd=%s\n", disconnect_cmd, reconnect_cmd);
    if (corrupt_at >= 0) printf("Config: checker self-test in iteration %d (%s)\n", corrupt_at, corrupt_nan ? "NaN" : "wrong value");
    printf("Config: every element of every iteration is checked on every rank; inputs are integers below %u\n", mask + 1);
  }

  if (myRank == 0) {
    printf("[Rank 0] Connecting NIC at start using: %s\n", reconnect_cmd);
    int rc = system(reconnect_cmd);
    if (rc != 0) {
      printf("[Rank 0] NIC connect command failed, rc=%d\n", rc);
    } else {
      printf("[Rank 0] NIC connect command completed.\n");
    }
  }
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  CUDACHECK(cudaSetDevice(localRank));

  ncclComm_t comm;
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  ncclUniqueId id;
  if (myRank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  printf("[Rank %d] Initializing NCCL Communicator...\n", myRank);
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));
  printf("[Rank %d] NCCL Initialization Complete.\n", myRank);

  size_t count = bytes / sizeof(float);
  if (count == 0) {
    if (myRank == 0) printf("Buffer size too small: %zu bytes\n", bytes);
    ncclCommDestroy(comm);
    CUDACHECK(cudaStreamDestroy(stream));
    MPICHECK(MPI_Finalize());
    return 1;
  }

  float* d_buf = nullptr;
  CUDACHECK(cudaMalloc(&d_buf, bytes));
  unsigned long long* d_stats = nullptr;   // [0] mismatching elements, [1] lowest mismatching index
  CUDACHECK(cudaMalloc(&d_stats, 2 * sizeof(unsigned long long)));

  int threads = 256;
  int blocks = (int)((count + threads - 1) / threads);

  bool overall_ok = true;
  unsigned long long local_mismatches = 0;
  std::vector<IbDevCounters> ib_devs;
  std::vector<std::vector<unsigned long long>> rx_log;
  std::vector<long long> iter_ms_log;
  std::vector<double> iter_t0, iter_t1;   // seconds since program start, rank 0
  if (myRank == 0) {
    ib_devs = init_ib_counters();
    rx_log.resize(iters);
    iter_ms_log.resize(iters, 0);
    iter_t0.resize(iters, 0);
    iter_t1.resize(iters, 0);
  }
  const auto prog_start = std::chrono::steady_clock::now();
  auto since_start = [&](std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(t - prog_start).count();
  };

  std::thread disconnect_thread;
  std::atomic<int> disconnect_rc{-1};
  double disconnect_t0 = -1, disconnect_t1 = -1;   // command start / end, seconds since program start
  if (myRank == 0 && start_disconnect_delay_ms >= 0) {
    printf("[Rank 0] Arming NIC disconnect at program start (delay %d ms) using: %s\n",
           start_disconnect_delay_ms, disconnect_cmd);
    disconnect_thread = std::thread([&, start_disconnect_delay_ms, disconnect_cmd]() {
      if (start_disconnect_delay_ms > 0) usleep((useconds_t)start_disconnect_delay_ms * 1000);
      disconnect_t0 = since_start(std::chrono::steady_clock::now());
      int rc = system(disconnect_cmd);
      disconnect_t1 = since_start(std::chrono::steady_clock::now());
      disconnect_rc = rc;
      if (rc != 0) {
        printf("[Rank 0] NIC disconnect command failed, rc=%d\n", rc);
      } else {
        printf("[Rank 0] NIC disconnect command completed.\n");
      }
    });
  }

  for (int iter = 0; iter < iters; ++iter) {
    auto iter_start = std::chrono::steady_clock::now();
    if (myRank == 0) {
      printf("[Rank 0] Iter %d/%d START: allreduce %.2f GiB\n", iter + 1, iters,
             (double)bytes / (1024.0 * 1024.0 * 1024.0));
    }

    fill_kernel<<<blocks, threads, 0, stream>>>(d_buf, count, myRank, iter, mask);
    CUDACHECK(cudaGetLastError());

    std::vector<unsigned long long> rx_start;
    if (myRank == 0) {
      rx_start.resize(ib_devs.size(), 0);
      for (size_t i = 0; i < ib_devs.size(); ++i) {
        if (!ib_devs[i].available) continue;
        if (!read_u64_file(ib_devs[i].rx_path, &rx_start[i])) {
          ib_devs[i].available = false;
        }
      }
    }

    NCCLCHECK(ncclAllReduce(d_buf, d_buf, count, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    auto iter_end = std::chrono::steady_clock::now();   // the reported time covers input fill and collective only

    if (myRank == 0) {
      rx_log[iter].resize(ib_devs.size(), 0);
      for (size_t i = 0; i < ib_devs.size(); ++i) {
        if (!ib_devs[i].available) continue;
        unsigned long long rx_now = 0;
        if (!read_u64_file(ib_devs[i].rx_path, &rx_now)) {
          ib_devs[i].available = false;
          continue;
        }
        long long rx_diff = (long long)rx_now - (long long)rx_start[i];
        if (rx_diff < 0) rx_diff = 0;
        rx_log[iter][i] = (unsigned long long)(rx_diff * 4 / 1024 / 1024);
      }
    }

    if (myRank == 0 && corrupt_at == iter + 1) {
      size_t idx = count / 3 + 12345;
      float bad = 0.0f;
      CUDACHECK(cudaMemcpy(&bad, d_buf + idx, sizeof(float), cudaMemcpyDeviceToHost));
      bad = corrupt_nan ? NAN : bad + 1.0f;
      CUDACHECK(cudaMemcpy(d_buf + idx, &bad, sizeof(float), cudaMemcpyHostToDevice));
      printf("[Rank 0] Checker self-test: wrote %s to element %zu of iteration %d\n", corrupt_nan ? "NaN" : "a wrong value",
             idx, iter + 1);
    }

    // Check every element, outside the timed part of the iteration.
    unsigned long long stats[2] = {0, ULLONG_MAX};
    CUDACHECK(cudaMemcpyAsync(d_stats, stats, sizeof(stats), cudaMemcpyHostToDevice, stream));
    check_kernel<<<blocks, threads, 0, stream>>>(d_buf, count, nRanks, iter, mask, d_stats, d_stats + 1);
    CUDACHECK(cudaGetLastError());
    CUDACHECK(cudaMemcpyAsync(stats, d_stats, sizeof(stats), cudaMemcpyDeviceToHost, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    bool iter_ok = stats[0] == 0;
    if (!iter_ok) {
      size_t idx = (size_t)stats[1];
      float got = 0.0f;
      CUDACHECK(cudaMemcpy(&got, d_buf + idx, sizeof(float), cudaMemcpyDeviceToHost));
      unsigned int expect = 0;
      for (int r = 0; r < nRanks; ++r) expect += input_value(idx, r, iter, mask);
      printf("[Rank %d] Iter %d: %llu of %zu elements wrong, first at index %zu: got=%f expected=%u diff=%f\n",
             myRank, iter + 1, stats[0], count, idx, got, expect, (double)got - expect);
      overall_ok = false;
      local_mismatches += stats[0];
    }

    if (myRank == 0) {
      auto iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start).count();
      iter_ms_log[iter] = (long long)iter_ms;
      iter_t0[iter] = since_start(iter_start);
      iter_t1[iter] = since_start(iter_end);
      printf("[Rank 0] Iter %d/%d END: %s (elapsed %lld ms)\n",
             iter + 1, iters, iter_ok ? "OK" : "FAIL", (long long)iter_ms);
    }
  }

  if (disconnect_thread.joinable()) {
    disconnect_thread.join();
  }

  if (myRank == 0) {
    printf("[Rank 0] Reconnecting NIC using: %s\n", reconnect_cmd);
    int rc = system(reconnect_cmd);
    if (rc != 0) {
      printf("[Rank 0] NIC reconnect command failed, rc=%d\n", rc);
    } else {
      printf("[Rank 0] NIC reconnect command completed.\n");
    }
  }

  if (myRank == 0 && !ib_devs.empty()) {
    printf("[Rank 0] IB RX per-iteration (MB, port_rcv_data *4B):\n");
    printf("%-6s %-10s", "Iter", "Time(ms)");
    for (size_t i = 0; i < ib_devs.size(); ++i) {
      std::string col = ib_devs[i].dev + "_RX";
      printf(" %-12s", col.c_str());
    }
    printf("\n");
    for (int iter = 0; iter < iters; ++iter) {
      printf("%-6d %-10lld", iter + 1, (long long)iter_ms_log[iter]);
      for (size_t i = 0; i < ib_devs.size(); ++i) {
        if (!ib_devs[i].available) {
          printf(" %-12s", "NA");
        } else {
          printf(" %-12llu", (unsigned long long)rx_log[iter][i]);
        }
      }
      printf("\n");
    }
  }

  int local_ok = overall_ok ? 1 : 0;
  int global_ok = 0;
  unsigned long long all_mismatches = 0;
  MPICHECK(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  MPICHECK(MPI_Reduce(&local_mismatches, &all_mismatches, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));

  // Exit code: 0 pass, 2 wrong results, 3 a NIC failure was injected but not observed during a collective
  // (1 is what the CUDA/NCCL/MPI error checks above exit with).
  int status = global_ok ? 0 : 2;
  if (myRank == 0) {
    printf("[Rank 0] Verification: all %zu elements of each of the %d iterations checked on all %d ranks: %llu wrong\n",
           count, iters, nRanks, all_mismatches);
    if (start_disconnect_delay_ms >= 0) {
      // The cut port is node-1's mlx5_2, the second counter. The iteration that was in flight when it was cut
      // received part of its share over it; the following ones receive nothing over it.
      const size_t port = 1;
      unsigned long long full = 0;
      for (int it = 0; it < iters; ++it) full = std::max(full, rx_log[it].size() > port ? rx_log[it][port] : 0ull);
      int cut = -1;
      for (int it = 0; it < iters && cut < 0; ++it) {
        if (rx_log[it].size() > port && rx_log[it][port] < full * 9 / 10) cut = it;
      }
      const unsigned long long idle_mb = 20;
      bool quiet_after = cut >= 0;
      for (int it = cut + 1; cut >= 0 && it < iters; ++it) quiet_after = quiet_after && rx_log[it][port] < idle_mb;
      char buf[512];
      if (!ib_devs[port].available) {
        snprintf(buf, sizeof(buf), "mlx5_2 counters unavailable");
        if (status == 0) status = 3;
      } else if (disconnect_rc != 0) {
        snprintf(buf, sizeof(buf), "disconnect command failed (rc=%d)", disconnect_rc.load());
        if (status == 0) status = 3;
      } else if (cut < 0) {
        snprintf(buf, sizeof(buf), "mlx5_2 kept its full share in every iteration: no failure observed");
        if (status == 0) status = 3;
      } else if (rx_log[cut][port] < idle_mb) {
        snprintf(buf, sizeof(buf), "mlx5_2 was already down when iteration %d started (%llu MB): the cut did not hit a collective in flight; run again",
                 cut + 1, rx_log[cut][port]);
        if (status == 0) status = 3;
      } else if (!quiet_after) {
        snprintf(buf, sizeof(buf), "mlx5_2 still carried traffic after iteration %d", cut + 1);
        if (status == 0) status = 3;
      } else {
        snprintf(buf, sizeof(buf), "mlx5_2 cut during iteration %d (%llu of %llu MB received before the cut; command ran %.2f-%.2f s, "
                 "iteration %.2f-%.2f s), no traffic on it in iterations %d-%d, all of them completed",
                 cut + 1, rx_log[cut][port], full, disconnect_t0, disconnect_t1, iter_t0[cut], iter_t1[cut], cut + 2, iters);
      }
      printf("[Rank 0] Failure evidence: %s\n", buf);
    }
    if (status == 0) {
      printf("[Rank 0] TEST PASS: all AllReduces completed and every element of every iteration is correct%s.\n",
             start_disconnect_delay_ms >= 0 ? ", including the one the NIC failure hit" : "");
    } else if (status == 2) {
      printf("[Rank 0] TEST FAIL: Verification failed on at least one rank/iteration.\n");
    } else {
      printf("[Rank 0] TEST FAIL: the NIC failure did not hit a running AllReduce (see Failure evidence above).\n");
    }
  }
  MPICHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));

  CUDACHECK(cudaFree(d_stats));
  CUDACHECK(cudaFree(d_buf));
  CUDACHECK(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  MPICHECK(MPI_Finalize());

  printf("[Rank %d] Exiting.\n", myRank);
  return status;
}
