#!/usr/bin/env bash
# nccl-tests on the healthy cluster with R2CC switched off (R2CC_MODE=0: this repository's library, all three
# NICs). The reference for the times relative to healthy.
#
# Usage: ./04.nccl_tests_baseline_healthy.sh [nccl-tests args...]
#   default args: -b 8 -e 4G -f 2 -g 1 -c 1 -n 5 -w 2 -d float -o sum
#   (-c 1: after the timed iterations of every size, one more AllReduce is compared element by element with the
#   expected result, which nccl-tests computes on the GPU; #wrong counts the differing elements)
#   NCCL_TEST_BIN=all_gather_perf ./04.nccl_tests_baseline_healthy.sh -b 64M -e 1G -f 2 -d float
#   Nothing is written to disk by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
check_idle
ensure_nic_connected
run_logged 04.nccl_tests_baseline_healthy run_nccl_tests 0 baseline_healthy "$@"
