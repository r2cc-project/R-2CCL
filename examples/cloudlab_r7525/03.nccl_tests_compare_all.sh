#!/usr/bin/env bash
# Runs the three nccl-tests scenarios back to back with identical arguments and prints one comparison table:
#   baseline_healthy           plain NCCL, all 3 NICs healthy      (04.nccl_tests_baseline_healthy.sh)
#   balance_unhealthy          R2CC-Balance, mlx5_2 failed         (05.nccl_tests_balance_unhealthy.sh)
#   r2cc_allreduce_unhealthy   R2CC-AllReduce, mlx5_2 failed       (06.nccl_tests_r2cc_allreduce_unhealthy.sh)
#
# Usage: ./03.nccl_tests_compare_all.sh [nccl-tests args...]
#   default args here: -b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum   (the full 8 B sweep takes ~2 min per scenario)
#   Nothing is written to disk by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EXAMPLE_DIR}/common.sh"
check_idle
ensure_nic_connected

main() {
  local ARGS=("$@")
  [[ ${#ARGS[@]} -eq 0 ]] && ARGS=(-b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum)
  TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT

  # Each scenario runs in a subshell so its R2CC_MODE cannot leak into the next one.
  ( source "${EXAMPLE_DIR}/common.sh"; NCCL_TESTS_RESULT_FILE="${TMP}/baseline_healthy" run_nccl_tests 0 baseline_healthy "${ARGS[@]}" )
  ( source "${EXAMPLE_DIR}/common.sh"; NCCL_TESTS_RESULT_FILE="${TMP}/balance_unhealthy" run_nccl_tests 2 balance_unhealthy "${ARGS[@]}" )
  ( source "${EXAMPLE_DIR}/common.sh"; NCCL_TESTS_RESULT_FILE="${TMP}/r2cc_allreduce_unhealthy" run_nccl_tests 3 r2cc_allreduce_unhealthy "${ARGS[@]}" )

  echo
  echo "===== comparison (${ARGS[*]}) ====="
  busbw_table \
    "baseline_healthy:${TMP}/baseline_healthy" \
    "balance_unhealthy:${TMP}/balance_unhealthy" \
    "r2cc_allreduce_unhealthy:${TMP}/r2cc_allreduce_unhealthy"
}
run_logged 03.nccl_tests_compare_all main "$@"
