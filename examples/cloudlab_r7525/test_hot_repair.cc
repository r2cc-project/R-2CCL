#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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

__global__ void fill_kernel(float* buf, size_t n, float val) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    buf[idx] = val;
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
  const int verify_samples = 4;
  const char* disconnect_cmd = "./nic/disconnect_nic1.sh";
  const char* reconnect_cmd = "./nic/connect_nic1.sh";

  const int start_disconnect_delay_ms = get_env_int("R2CC_AR_START_DISCONNECT_DELAY_MS", -1);

  if (myRank == 0) {
    printf("Config: iters=%d, bytes=%zu (%.2f GiB), count=%zu floats\n",
           iters, bytes, (double)bytes / (1024.0 * 1024.0 * 1024.0), bytes / sizeof(float));
    printf("Config: start_disconnect_delay_ms=%d\n", start_disconnect_delay_ms);
    printf("Config: disconnect_cmd=%s, reconnect_cmd=%s\n", disconnect_cmd, reconnect_cmd);
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

  int threads = 256;
  int blocks = (int)((count + threads - 1) / threads);

  bool overall_ok = true;
  std::vector<IbDevCounters> ib_devs;
  std::vector<std::vector<unsigned long long>> rx_log;
  std::vector<long long> iter_ms_log;
  if (myRank == 0) {
    ib_devs = init_ib_counters();
    rx_log.resize(iters);
    iter_ms_log.resize(iters, 0);
  }

  std::thread disconnect_thread;
  if (myRank == 0 && start_disconnect_delay_ms >= 0) {
    printf("[Rank 0] Arming NIC disconnect at program start (delay %d ms) using: %s\n",
           start_disconnect_delay_ms, disconnect_cmd);
    disconnect_thread = std::thread([start_disconnect_delay_ms, disconnect_cmd]() {
      if (start_disconnect_delay_ms > 0) usleep((useconds_t)start_disconnect_delay_ms * 1000);
      int rc = system(disconnect_cmd);
      if (rc != 0) {
        printf("[Rank 0] NIC disconnect command failed, rc=%d\n", rc);
      } else {
        printf("[Rank 0] NIC disconnect command completed.\n");
      }
    });
  }

  for (int iter = 0; iter < iters; ++iter) {
    float base = (float)(iter + 1);
    float in_val = base + (float)myRank;
    double expected = (double)base * nRanks + ((double)nRanks * (nRanks - 1) / 2.0);

    auto iter_start = std::chrono::steady_clock::now();
    if (myRank == 0) {
      printf("[Rank 0] Iter %d/%d START: allreduce %.2f GiB\n", iter + 1, iters,
             (double)bytes / (1024.0 * 1024.0 * 1024.0));
    }

    fill_kernel<<<blocks, threads, 0, stream>>>(d_buf, count, in_val);
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

    bool iter_ok = true;
    if (verify_samples > 0) {
      std::vector<size_t> samples;
      samples.push_back(0);
      if (count > 1) samples.push_back(count / 2);
      if (count > 2) samples.push_back(count - 1);
      while ((int)samples.size() < verify_samples) {
        size_t idx = (samples.back() * 1315423911u + 12345u) % count;
        samples.push_back(idx);
      }

      for (size_t idx : samples) {
        float host_val = 0.0f;
        CUDACHECK(cudaMemcpy(&host_val, d_buf + idx, sizeof(float), cudaMemcpyDeviceToHost));
        double diff = fabs((double)host_val - expected);
        if (diff > 1e-3) {
          printf("[Rank %d] Iter %d sample idx=%zu got=%f expected=%f diff=%f\n",
                 myRank, iter + 1, idx, host_val, (float)expected, (float)diff);
          iter_ok = false;
          break;
        }
      }
    }

    if (!iter_ok) {
      overall_ok = false;
    }

    if (myRank == 0) {
      auto iter_end = std::chrono::steady_clock::now();
      auto iter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start).count();
      iter_ms_log[iter] = (long long)iter_ms;
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
  MPICHECK(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));

  if (myRank == 0) {
    if (global_ok) {
      printf("[Rank 0] TEST PASS: All allreduces completed and verified.\n");
    } else {
      printf("[Rank 0] TEST FAIL: Verification failed on at least one rank/iteration.\n");
    }
  }

  CUDACHECK(cudaFree(d_buf));
  CUDACHECK(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  MPICHECK(MPI_Finalize());

  printf("[Rank %d] Exiting.\n", myRank);
  return 0;
}
