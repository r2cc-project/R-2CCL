// The hot-repair test of 01/02 (run_hot_repair.sh): 10 AllReduces of 4 GiB floats on all ranks, node-1's mlx5_2 cut on
// the SmartNIC R2CC_AR_START_DISCONNECT_DELAY_MS after the start, and every element of every iteration checked on every
// rank (test_common.h). Exit code: 0 pass, 2 wrong results, 3 the cut did not hit a running AllReduce, 1 a
// CUDA/NCCL/MPI error.
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "test_common.h"

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
  const CorruptTest corrupt;
  const unsigned int mask = input_mask(nRanks);

  if (myRank == 0) {
    printf("Config: iters=%d, bytes=%zu (%.2f GiB), count=%zu floats\n",
           iters, bytes, (double)bytes / (1024.0 * 1024.0 * 1024.0), bytes / sizeof(float));
    printf("Config: start_disconnect_delay_ms=%d\n", start_disconnect_delay_ms);
    printf("Config: disconnect_cmd=%s, reconnect_cmd=%s\n", disconnect_cmd, reconnect_cmd);
    if (corrupt.at >= 0) printf("Config: checker self-test in iteration %d (%s)\n", corrupt.at, corrupt.nan ? "NaN" : "wrong value");
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
  unsigned long long* d_stats = nullptr;   // scratch of check_output()
  CUDACHECK(cudaMalloc(&d_stats, 2 * sizeof(unsigned long long)));

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

    fill_input(d_buf, count, myRank, iter, mask, stream);

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

    corrupt.apply(d_buf, count, myRank, iter + 1);

    // Check every element, outside the timed part of the iteration.
    char what[32];
    snprintf(what, sizeof(what), "Iter %d", iter + 1);
    unsigned long long wrong = check_output(d_buf, count, myRank, nRanks, iter, mask, d_stats, stream, what);
    bool iter_ok = wrong == 0;
    if (!iter_ok) {
      overall_ok = false;
      local_mismatches += wrong;
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
      const size_t port = 1;   // node-1's mlx5_2, the port that is cut
      std::vector<unsigned long long> rx_mb(iters, 0);
      for (int it = 0; it < iters; ++it) rx_mb[it] = rx_log[it].size() > port ? rx_log[it][port] : 0ull;
      char buf[512];
      int evidence = failure_evidence(rx_mb, ib_devs[port].available, disconnect_rc.load(), disconnect_t0,
                                      disconnect_t1, iter_t0, iter_t1, buf, sizeof(buf));
      if (status == 0) status = evidence;
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
