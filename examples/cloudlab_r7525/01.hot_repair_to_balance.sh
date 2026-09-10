#!/usr/bin/env bash
# Real NIC failure -> hot repair -> R2CC-Balance (the demo scenario).
#
#   hot_repair/test_hot_repair runs 10 x 4 GiB float AllReduce on the 4 GPUs. 4 s after start the SmartNIC
#   drops all traffic of node-1's mlx5_2 (nic/disconnect_nic1.sh). R2CC detects the failure mid-collective,
#   live-migrates the in-flight transfers to the backup connection, and from the next AllReduce on
#   (R2CC_AR_AFTER_REPAIR=2) re-balances all traffic over the remaining NICs. Every iteration is verified;
#   the final table shows per-iteration time and per-HCA RX bytes (mlx5_2 goes to 0 after failover).
#   The NIC is reconnected at the end. Expect: TEST PASS.
#
# Usage: ./01.hot_repair_to_balance.sh [-log 0|1]        (-log 1 = keep NCCL INFO R2CC trace lines)
#   Nothing is written to disk by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${EXAMPLE_DIR}"
check_idle
export R2CC_AR_AFTER_REPAIR=2                                  # 2 = Balance after the repair
run_logged 01.hot_repair_to_balance ./hot_repair/run_hot_repair.sh "$@"
