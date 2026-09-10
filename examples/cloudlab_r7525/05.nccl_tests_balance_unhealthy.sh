#!/usr/bin/env bash
# nccl-tests, R2CC-Balance on the degraded cluster (R2CC_MODE=2).
#
#   No NIC is cut. node-1's mlx5_2 is declared failed (R2CC_FAILED_NODE / R2CC_FAILED_HCA from common.sh); the
#   channels whose ring hop would use it are excluded on every rank and the collective runs over the
#   remaining NICs, exactly as after a real hot repair. Results are checked against a CPU reference (-c 1).
#
# Usage: ./05.nccl_tests_balance_unhealthy.sh [nccl-tests args...]
#   default args: -b 8 -e 4G -f 2 -g 1 -c 1 -n 5 -w 2 -d float -o sum
#   other collectives: NCCL_TEST_BIN=all_gather_perf ./05.nccl_tests_balance_unhealthy.sh -b 64M -e 1G -f 2 -d float
#   Nothing is written to disk by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
check_idle
ensure_nic_connected
run_logged 05.nccl_tests_balance_unhealthy run_nccl_tests 2 balance_unhealthy "$@"
