#!/usr/bin/env bash
# nccl-tests, R2CC-AllReduce on the degraded cluster (R2CC_MODE=3).
#
#   Same degraded topology as 05 (node-1's mlx5_2 declared failed, X = 1/3), but AllReduce runs the two-stage
#   algorithm: AllReduce of the first (1-X) on all ranks, the partial AllReduce of the last X on the healthy
#   node's sub-communicator, then a K-way pipelined Reduce of the tail to a helper rank on the healthy node
#   and Broadcast of each chunk to everyone. Sub-communicators are created on the first eligible call
#   (>= R2CC_AR_MIN_BYTES, default 16 MiB; smaller messages fall back to Balance). Results are checked with -c 1.
#
# Usage: ./06.nccl_tests_r2cc_allreduce_unhealthy.sh [nccl-tests args...]
#   default args: -b 8 -e 4G -f 2 -g 1 -c 1 -n 5 -w 2 -d float -o sum
#   knobs (env): R2CC_AR_STAGE2_CHUNKS (K, default 4), R2CC_AR_SCHEDULE (0 all concurrent, 1 Stage 2 after
#                Stage 1, 2 (default) also tail AllReduce after main AllReduce, 3 also serialize Reduce/Broadcast
#                chunks), R2CC_AR_MIN_BYTES.
#   Nothing is written to disk by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
check_idle
ensure_nic_connected
run_logged 06.nccl_tests_r2cc_allreduce_unhealthy run_nccl_tests 3 r2cc_allreduce_unhealthy "$@"
