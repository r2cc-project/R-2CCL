#!/usr/bin/env bash
# Real NIC failure -> hot repair -> R2CC-AllReduce.
#
#   Same run as 01 (10 x 4 GiB AllReduce, node-1's mlx5_2 cut at t=4 s, hot repair of the in-flight
#   collective), but with R2CC_AR_AFTER_REPAIR=3 the library switches to R2CC-AllReduce afterwards:
#     Stage 1  AllReduce of the first (1-X) of the data on all ranks (Balance-masked channels), then the
#              partial AllReduce of the last X on the healthy node's sub-communicator;
#     Stage 2  K-way pipelined Reduce of the tail from the degraded node to a helper rank on the healthy node
#              (its input is the partial result) and Broadcast of each finished chunk back to everyone.
#   X = failed NICs / NICs per node (1/3 here). The first AllReduce after the switch also creates the two
#   sub-communicators (slower once). Expect: TEST PASS; mlx5_2 RX 0 after failover.
#
# Usage: ./02.hot_repair_to_r2cc_allreduce.sh [-log 0|1]
#   Tunables (env): R2CC_AR_STAGE2_CHUNKS (K, default 4), R2CC_AR_SCHEDULE (0-3, default 2 = Stage 1 serialized),
#   R2CC_AR_MIN_BYTES (default 16 MiB). Nothing is written to disk by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${EXAMPLE_DIR}"
check_idle
export R2CC_AR_AFTER_REPAIR=3                                  # 3 = R2CC-AllReduce after the repair
run_logged 02.hot_repair_to_r2cc_allreduce ./hot_repair/run_hot_repair.sh "$@"
