#!/usr/bin/env bash
# Clone (if needed) and build nccl-tests against this repository's NCCL build, then copy it to node-2 at the
# same path so mpirun can start the same binary on both nodes. Used by test3/test4.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/mydata/nccl-tests}"
REMOTE_HOST="${REMOTE_HOST:-node-2}"
MPI_HOME="${MPI_HOME:-/mydata/openMpi}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_70,code=sm_70}"   # V100

if [[ ! -d "${NCCL_TESTS_DIR}/.git" ]]; then
  git clone https://github.com/NVIDIA/nccl-tests.git "${NCCL_TESTS_DIR}"
fi
make -C "${NCCL_TESTS_DIR}" -j"$(nproc)" MPI=1 MPI_HOME="${MPI_HOME}" NCCL_HOME="${REPO_ROOT}/build" \
     CUDA_HOME="${CUDA_HOME}" NVCC_GENCODE="${NVCC_GENCODE}"
echo "Copying ${NCCL_TESTS_DIR} to ${REMOTE_HOST}:${NCCL_TESTS_DIR}"
rsync -a "${NCCL_TESTS_DIR}/" "${REMOTE_HOST}:${NCCL_TESTS_DIR}/"
echo "Done: ${NCCL_TESTS_DIR}/build/all_reduce_perf"
