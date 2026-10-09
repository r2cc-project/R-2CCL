// Times a 4 GiB float AllReduce captured in a CUDA graph and replayed in every iteration, while node-1's mlx5_2 is
// cut on the SmartNIC 4 s after the start (08.hot_repair_cuda_graph.sh). Run from the example root on all ranks:
//   graph_timing                                  replay the graph captured at the start in every iteration
//   graph_timing --recapture-at 6                 capture the AllReduce again after iteration 6
//   graph_timing --recapture-at 6 --eager         ... after one AllReduce outside the capture
// Rank 0 prints the cost of every capture and, at the end, the time and the MB received per NIC of every iteration.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "mpi.h"

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

static const char* kDevs[] = {"mlx5_0", "mlx5_2", "mlx5_3"};
static const int kNumDevs = 3;

// Bytes received by every NIC so far (port_rcv_data counts 4-byte words); 0 where the counter is missing.
static void read_rx(unsigned long long* rx) {
  for (int i = 0; i < kNumDevs; ++i) {
    std::string path = std::string("/sys/class/infiniband/") + kDevs[i] + "/ports/1/counters/port_rcv_data";
    unsigned long long v = 0;
    FILE* f = fopen(path.c_str(), "r");
    if (f) {
      if (fscanf(f, "%llu", &v) != 1) v = 0;
      fclose(f);
    }
    rx[i] = v * 4;
  }
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char* argv[]) {
  setvbuf(stdout, NULL, _IOLBF, 0);
  int recapture_at = -1, eager = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--recapture-at") && i + 1 < argc) recapture_at = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--eager")) eager = 1;
  }

  int myRank = 0, nRanks = 0;
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));
  const char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  const int localRank = lr ? atoi(lr) : myRank % 2;

  const size_t bytes = 4ull * 1024 * 1024 * 1024;
  const size_t count = bytes / sizeof(float);
  const int iters = 10;
  const char* env_delay = getenv("R2CC_AR_START_DISCONNECT_DELAY_MS");
  const int delay_ms = env_delay ? atoi(env_delay) : 4000;
  if (myRank == 0) {
    printf("Config: ranks=%d iters=%d bytes=%zu disconnect_delay_ms=%d recapture_at=%d eager=%d\n", nRanks, iters,
           bytes, delay_ms, recapture_at, eager);
    if (system("./nic/connect_nic1.sh") != 0) printf("[Rank 0] NIC connect command failed\n");
  }
  MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

  CUDACHECK(cudaSetDevice(localRank));
  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));
  ncclUniqueId id;
  if (myRank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

  float* buf = nullptr;
  CUDACHECK(cudaMalloc(&buf, bytes));
  CUDACHECK(cudaMemset(buf, 0, bytes));

  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  auto capture = [&]() {
    if (exec) CUDACHECK(cudaGraphExecDestroy(exec));
    if (graph) CUDACHECK(cudaGraphDestroy(graph));
    auto t0 = std::chrono::steady_clock::now();
    CUDACHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    NCCLCHECK(ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
    CUDACHECK(cudaStreamEndCapture(stream, &graph));
    double capture_ms = ms_since(t0);
    t0 = std::chrono::steady_clock::now();
    CUDACHECK(cudaGraphInstantiate(&exec, graph, NULL, NULL, 0));
    double instantiate_ms = ms_since(t0);
    size_t nodes = 0;
    CUDACHECK(cudaGraphGetNodes(graph, NULL, &nodes));
    if (myRank == 0) {
      printf("[Rank 0] Captured AllReduce into a CUDA graph: capture %.1f ms, instantiate %.1f ms, %zu nodes\n",
             capture_ms, instantiate_ms, nodes);
    }
  };
  capture();

  std::thread cut;
  if (myRank == 0 && delay_ms >= 0) {
    cut = std::thread([delay_ms]() {
      usleep((useconds_t)delay_ms * 1000);
      int rc = system("./nic/disconnect_nic1.sh");
      printf("[Rank 0] NIC disconnect command %s\n", rc == 0 ? "completed" : "failed");
    });
  }

  std::vector<double> iter_ms(iters, 0);
  std::vector<unsigned long long> rx_mb(iters * kNumDevs, 0);
  for (int it = 0; it < iters; ++it) {
    unsigned long long rx0[kNumDevs], rx1[kNumDevs];
    if (myRank == 0) read_rx(rx0);
    auto t0 = std::chrono::steady_clock::now();
    CUDACHECK(cudaGraphLaunch(exec, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    iter_ms[it] = ms_since(t0);
    if (myRank == 0) {
      read_rx(rx1);
      for (int d = 0; d < kNumDevs; ++d) rx_mb[it * kNumDevs + d] = (rx1[d] - rx0[d]) / (1024 * 1024);
      printf("[Rank 0] Iter %d/%d: %.0f ms\n", it + 1, iters, iter_ms[it]);
    }
    if (recapture_at == it + 1) {
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      if (eager) {
        auto e0 = std::chrono::steady_clock::now();
        NCCLCHECK(ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        if (myRank == 0) printf("[Rank 0] Eager AllReduce before re-capture took %.0f ms\n", ms_since(e0));
      }
      capture();
    }
  }

  if (cut.joinable()) cut.join();
  if (myRank == 0) {
    if (system("./nic/connect_nic1.sh") != 0) printf("[Rank 0] NIC reconnect command failed\n");
    printf("Iter   Time(ms)   mlx5_0_RX    mlx5_2_RX    mlx5_3_RX\n");
    for (int it = 0; it < iters; ++it) {
      printf("%-6d %-10.0f %-12llu %-12llu %-12llu\n", it + 1, iter_ms[it], rx_mb[it * kNumDevs],
             rx_mb[it * kNumDevs + 1], rx_mb[it * kNumDevs + 2]);
    }
  }

  CUDACHECK(cudaGraphExecDestroy(exec));
  CUDACHECK(cudaGraphDestroy(graph));
  CUDACHECK(cudaFree(buf));
  CUDACHECK(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  MPICHECK(MPI_Finalize());
  return 0;
}
