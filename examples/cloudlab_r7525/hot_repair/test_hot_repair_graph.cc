// The hot repair of 01/02 with the AllReduce captured in a CUDA graph (08.hot_repair_cuda_graph.sh): a 4 GiB float
// AllReduce on all ranks is captured once and replayed in each of 10 iterations, while node-1's mlx5_2 is cut on the
// SmartNIC 4 s after the start. Run from the example root on all ranks:
//   test_hot_repair_graph                                  replay the graph captured at the start in every iteration
//   test_hot_repair_graph --recapture-at 6                 capture the AllReduce again after iteration 6
//   test_hot_repair_graph --recapture-at 6 --eager         ... after one AllReduce outside the capture
// Every iteration gets new inputs, and every element of every AllReduce output (the replays and the AllReduce outside
// the capture) is checked on every rank, as in 01/02 (test_common.h); the inputs are written and the outputs checked
// outside the timed part. Rank 0 prints the cost of every capture and, at the end, the time and the MB received per NIC
// of every iteration, the result of the check and the evidence that the cut hit a running AllReduce. Exit code: 0 pass,
// 2 wrong results, 3 the cut did not hit a running AllReduce, 1 a CUDA/NCCL/MPI error.
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>

#include "test_common.h"

static const char* kDevs[] = {"mlx5_0", "mlx5_2", "mlx5_3"};
static const int kNumDevs = 3;

// Bytes received by every NIC so far (port_rcv_data counts 4-byte words); 0 where the counter is missing, and then
// the return value is false.
static bool read_rx(unsigned long long* rx) {
  bool all = true;
  for (int i = 0; i < kNumDevs; ++i) {
    std::string path = std::string("/sys/class/infiniband/") + kDevs[i] + "/ports/1/counters/port_rcv_data";
    unsigned long long v = 0;
    FILE* f = fopen(path.c_str(), "r");
    if (!f || fscanf(f, "%llu", &v) != 1) {
      v = 0;
      all = false;
    }
    if (f) fclose(f);
    rx[i] = v * 4;
  }
  return all;
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
  const CorruptTest corrupt;
  const unsigned int mask = input_mask(nRanks);
  if (myRank == 0) {
    printf("Config: ranks=%d iters=%d bytes=%zu disconnect_delay_ms=%d recapture_at=%d eager=%d\n", nRanks, iters,
           bytes, delay_ms, recapture_at, eager);
    if (corrupt.at >= 0) printf("Config: checker self-test in iteration %d (%s)\n", corrupt.at, corrupt.nan ? "NaN" : "wrong value");
    printf("Config: every element of every AllReduce is checked on every rank; inputs are integers below %u\n", mask + 1);
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
  unsigned long long* d_stats = nullptr;   // scratch of check_output()
  CUDACHECK(cudaMalloc(&d_stats, 2 * sizeof(unsigned long long)));
  unsigned long long wrong = 0;            // wrong elements on this rank, over all checked AllReduces

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

  const auto prog_start = std::chrono::steady_clock::now();
  auto since_start = [&]() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - prog_start).count();
  };
  std::thread cut;
  std::atomic<int> cut_rc{-1};
  double cut_t0 = -1, cut_t1 = -1;   // disconnect command start / end, seconds since the start
  if (myRank == 0 && delay_ms >= 0) {
    cut = std::thread([&, delay_ms]() {
      usleep((useconds_t)delay_ms * 1000);
      cut_t0 = since_start();
      int rc = system("./nic/disconnect_nic1.sh");
      cut_t1 = since_start();
      cut_rc = rc;
      printf("[Rank 0] NIC disconnect command %s\n", rc == 0 ? "completed" : "failed");
    });
  }

  std::vector<double> iter_ms(iters, 0), iter_t0(iters, 0), iter_t1(iters, 0);
  std::vector<unsigned long long> rx_mb(iters * kNumDevs, 0);
  bool counters = true;
  for (int it = 0; it < iters; ++it) {
    fill_input(buf, count, myRank, it, mask, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
    unsigned long long rx0[kNumDevs], rx1[kNumDevs];
    if (myRank == 0) counters = read_rx(rx0) && counters;
    iter_t0[it] = since_start();
    auto t0 = std::chrono::steady_clock::now();
    CUDACHECK(cudaGraphLaunch(exec, stream));
    CUDACHECK(cudaStreamSynchronize(stream));
    iter_ms[it] = ms_since(t0);
    iter_t1[it] = since_start();
    if (myRank == 0) {
      counters = read_rx(rx1) && counters;
      for (int d = 0; d < kNumDevs; ++d) rx_mb[it * kNumDevs + d] = (rx1[d] - rx0[d]) / (1024 * 1024);
      printf("[Rank 0] Iter %d/%d: %.0f ms\n", it + 1, iters, iter_ms[it]);
    }
    corrupt.apply(buf, count, myRank, it + 1);
    char what[32];
    snprintf(what, sizeof(what), "Iter %d", it + 1);
    wrong += check_output(buf, count, myRank, nRanks, it, mask, d_stats, stream, what);

    if (recapture_at == it + 1) {
      MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
      if (eager) {
        fill_input(buf, count, myRank, iters, mask, stream);   // inputs of none of the replayed iterations
        CUDACHECK(cudaStreamSynchronize(stream));
        auto e0 = std::chrono::steady_clock::now();
        NCCLCHECK(ncclAllReduce(buf, buf, count, ncclFloat, ncclSum, comm, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        if (myRank == 0) printf("[Rank 0] Eager AllReduce before re-capture took %.0f ms\n", ms_since(e0));
        wrong += check_output(buf, count, myRank, nRanks, iters, mask, d_stats, stream, "AllReduce before re-capture");
      }
      capture();
    }
  }

  if (cut.joinable()) cut.join();
  unsigned long long all_wrong = 0;
  MPICHECK(MPI_Allreduce(&wrong, &all_wrong, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD));
  int status = all_wrong == 0 ? 0 : 2;
  if (myRank == 0) {
    if (system("./nic/connect_nic1.sh") != 0) printf("[Rank 0] NIC reconnect command failed\n");
    printf("Iter   Time(ms)   mlx5_0_RX    mlx5_2_RX    mlx5_3_RX\n");
    for (int it = 0; it < iters; ++it) {
      printf("%-6d %-10.0f %-12llu %-12llu %-12llu\n", it + 1, iter_ms[it], rx_mb[it * kNumDevs],
             rx_mb[it * kNumDevs + 1], rx_mb[it * kNumDevs + 2]);
    }
    printf("[Rank 0] Verification: all %zu elements of each of the %d iterations%s checked on all %d ranks: %llu wrong\n",
           count, iters, eager && recapture_at >= 1 && recapture_at <= iters ? " and of the AllReduce before re-capture" : "",
           nRanks, all_wrong);
    if (delay_ms >= 0) {
      std::vector<unsigned long long> rx2(iters, 0);   // node-1's mlx5_2, the port that is cut
      for (int it = 0; it < iters; ++it) rx2[it] = rx_mb[it * kNumDevs + 1];
      char msg[512];
      int evidence = failure_evidence(rx2, counters, cut_rc.load(), cut_t0, cut_t1, iter_t0, iter_t1, msg, sizeof(msg));
      if (status == 0) status = evidence;
      printf("[Rank 0] Failure evidence: %s\n", msg);
    }
    if (status == 0) {
      printf("[Rank 0] TEST PASS: all AllReduces completed and every element of every iteration is correct%s.\n",
             delay_ms >= 0 ? ", including the one the NIC failure hit" : "");
    } else if (status == 2) {
      printf("[Rank 0] TEST FAIL: Verification failed on at least one rank/iteration.\n");
    } else {
      printf("[Rank 0] TEST FAIL: the NIC failure did not hit a running AllReduce (see Failure evidence above).\n");
    }
  }
  MPICHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));

  CUDACHECK(cudaGraphExecDestroy(exec));
  CUDACHECK(cudaGraphDestroy(graph));
  CUDACHECK(cudaFree(d_stats));
  CUDACHECK(cudaFree(buf));
  CUDACHECK(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  MPICHECK(MPI_Finalize());
  return status;
}
