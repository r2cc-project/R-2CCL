// Shared by test_hot_repair.cc (tests 01/02) and test_hot_repair_graph.cc (test 08): error checks, the inputs and the
// element-by-element check of the AllReduce output, the checker self-test, and the evidence that the NIC failure hit
// a running AllReduce.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <climits>
#include <cmath>
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

static const int kThreads = 256;

// Inputs are integers below 2^(24 - rank_bits); the sum over nRanks <= 2^rank_bits ranks stays below 2^24.
static unsigned int input_mask(int nRanks) {
  int rank_bits = 0;
  while ((1 << rank_bits) < nRanks) ++rank_bits;
  return (1u << (24 - rank_bits)) - 1;
}

static void fill_input(float* buf, size_t count, int rank, int iter, unsigned int mask, cudaStream_t stream) {
  fill_kernel<<<(unsigned)((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(buf, count, rank, iter, mask);
  CUDACHECK(cudaGetLastError());
}

// Checks every element of the output for input iteration iter (d_stats: two device words of scratch) and returns the
// number of wrong elements. A wrong output is reported with the count and the first wrong element, after "what".
static unsigned long long check_output(const float* buf, size_t count, int myRank, int nRanks, int iter,
                                       unsigned int mask, unsigned long long* d_stats, cudaStream_t stream,
                                       const char* what) {
  unsigned long long stats[2] = {0, ULLONG_MAX};   // mismatching elements, lowest mismatching index
  CUDACHECK(cudaMemcpyAsync(d_stats, stats, sizeof(stats), cudaMemcpyHostToDevice, stream));
  check_kernel<<<(unsigned)((count + kThreads - 1) / kThreads), kThreads, 0, stream>>>(buf, count, nRanks, iter, mask,
                                                                                     d_stats, d_stats + 1);
  CUDACHECK(cudaGetLastError());
  CUDACHECK(cudaMemcpyAsync(stats, d_stats, sizeof(stats), cudaMemcpyDeviceToHost, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  if (stats[0] != 0) {
    size_t idx = (size_t)stats[1];
    float got = 0.0f;
    CUDACHECK(cudaMemcpy(&got, buf + idx, sizeof(float), cudaMemcpyDeviceToHost));
    unsigned int expect = 0;
    for (int r = 0; r < nRanks; ++r) expect += input_value(idx, r, iter, mask);
    printf("[Rank %d] %s: %llu of %zu elements wrong, first at index %zu: got=%f expected=%u diff=%f\n",
           myRank, what, stats[0], count, idx, got, expect, (double)got - expect);
  }
  return stats[0];
}

// Checker self-test: R2CC_TEST_CORRUPT=<n>[,nan] overwrites one output element of rank 0 in iteration n (a wrong
// value, or NaN) before it is checked. The run must then end in TEST FAIL with a non-zero exit code.
struct CorruptTest {
  int at = -1, nan = 0;
  CorruptTest() {
    if (const char* c = getenv("R2CC_TEST_CORRUPT")) {
      at = atoi(c);
      nan = strstr(c, "nan") != nullptr;
    }
  }
  void apply(float* buf, size_t count, int myRank, int iter) const {   // iter counts from 1
    if (myRank != 0 || iter != at) return;
    size_t idx = count / 3 + 12345;
    float bad = 0.0f;
    CUDACHECK(cudaMemcpy(&bad, buf + idx, sizeof(float), cudaMemcpyDeviceToHost));
    bad = nan ? NAN : bad + 1.0f;
    CUDACHECK(cudaMemcpy(buf + idx, &bad, sizeof(float), cudaMemcpyHostToDevice));
    printf("[Rank 0] Checker self-test: wrote %s to element %zu of iteration %d\n", nan ? "NaN" : "a wrong value",
           idx, iter);
  }
};

// Whether the cut of node-1's mlx5_2 hit a running AllReduce, from the MB the port received in every iteration
// (rx_mb), the disconnect command and the time span of every iteration (seconds since the start). The iteration that
// was in flight when the port was cut received part of its share over it; the following ones receive nothing over it.
// Writes the text of the "Failure evidence" line to msg and returns 0, or 3 if the failure was not observed during a
// collective.
static int failure_evidence(const std::vector<unsigned long long>& rx_mb, bool counters, int disconnect_rc,
                            double cmd_t0, double cmd_t1, const std::vector<double>& iter_t0,
                            const std::vector<double>& iter_t1, char* msg, size_t len) {
  const int iters = (int)rx_mb.size();
  const unsigned long long idle_mb = 20;
  unsigned long long full = 0;
  for (int it = 0; it < iters; ++it) full = std::max(full, rx_mb[it]);
  int cut = -1;
  for (int it = 0; it < iters && cut < 0; ++it) {
    if (rx_mb[it] < full * 9 / 10) cut = it;
  }
  bool quiet_after = cut >= 0;
  for (int it = cut + 1; cut >= 0 && it < iters; ++it) quiet_after = quiet_after && rx_mb[it] < idle_mb;
  if (!counters) {
    snprintf(msg, len, "mlx5_2 counters unavailable");
  } else if (disconnect_rc != 0) {
    snprintf(msg, len, "disconnect command failed (rc=%d)", disconnect_rc);
  } else if (cut < 0) {
    snprintf(msg, len, "mlx5_2 kept its full share in every iteration: no failure observed");
  } else if (rx_mb[cut] < idle_mb) {
    snprintf(msg, len, "mlx5_2 was already down when iteration %d started (%llu MB): the cut did not hit a collective in flight; run again",
             cut + 1, rx_mb[cut]);
  } else if (!quiet_after) {
    snprintf(msg, len, "mlx5_2 still carried traffic after iteration %d", cut + 1);
  } else {
    snprintf(msg, len, "mlx5_2 cut during iteration %d (%llu of %llu MB received before the cut; command ran %.2f-%.2f s, "
             "iteration %.2f-%.2f s), no traffic on it in iterations %d-%d, all of them completed",
             cut + 1, rx_mb[cut], full, cmd_t0, cmd_t1, iter_t0[cut], iter_t1[cut], cut + 2, iters);
    return 0;
  }
  return 3;
}
