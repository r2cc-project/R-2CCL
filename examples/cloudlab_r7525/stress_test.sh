#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./stress_test.sh -n <rounds>

Options:
  -n <rounds>   Number of runs.
  -h, --help    Show this help.
EOF
}

extract_failover_iter() {
  local run_log="$1"
  awk '
    BEGIN {
      in_table = 0;
      col = -1;
    }
    /^Iter[[:space:]]+Time\(ms\)/ {
      in_table = 1;
      col = -1;
      for (i = 1; i <= NF; i++) {
        if ($i == "mlx5_2_RX") {
          col = i;
          break;
        }
      }
      next;
    }
    in_table {
      if (NF == 0) {
        in_table = 0;
        next;
      }
      if ($1 !~ /^[0-9]+$/) {
        in_table = 0;
        next;
      }
      if (col > 0 && col <= NF) {
        v = $(col) + 0;
        if (v < 20) {
          print $1;
          exit;
        }
      }
      next;
    }
  ' "${run_log}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST_CMD="${SCRIPT_DIR}/run_hot_repair.sh"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/stress}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

if [[ ! -x "${TEST_CMD}" ]]; then
  echo "test command not found or not executable: ${TEST_CMD}"
  exit 1
fi

rounds=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)
      if [[ $# -lt 2 ]]; then
        usage
        exit 1
      fi
      rounds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${rounds}" || ! "${rounds}" =~ ^[0-9]+$ || "${rounds}" -le 0 ]]; then
  echo "Invalid -n value: ${rounds:-<empty>}"
  usage
  exit 1
fi

mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/stress_summary_${RUN_TAG}.log"

pass_count=0
diff_count=0
segfault_count=0
timeout_count=0
unknown_count=0
declare -A failover_hist=()

for ((i=1; i<=rounds; i++)); do
  run_log="${LOG_DIR}/run_${RUN_TAG}_$(printf '%03d' "${i}").log"
  echo "[stress] run ${i}/${rounds}: timeout=2m log=${run_log}" | tee -a "${SUMMARY_LOG}"

  set +e
  (
    cd "${SCRIPT_DIR}"
    timeout 2m ./run_hot_repair.sh
  ) > "${run_log}" 2>&1
  rc=$?
  set -e

  result="unknown"
  if grep -qiE 'Segmentation fault|Caught signal 11|exited on signal 11|Signal: Segmentation fault' "${run_log}"; then
    result="segment_fault"
    ((segfault_count+=1))
  elif grep -q 'diff=' "${run_log}" || grep -q 'TEST FAIL: Verification failed' "${run_log}"; then
    result="diff"
    ((diff_count+=1))
  elif grep -q 'TEST PASS' "${run_log}"; then
    result="pass"
    ((pass_count+=1))
  elif [[ "${rc}" -eq 124 ]]; then
    result="timeout"
    ((timeout_count+=1))
  else
    ((unknown_count+=1))
  fi

  failover_iter="$(extract_failover_iter "${run_log}")"
  if [[ -z "${failover_iter}" ]]; then
    failover_iter="NA"
  fi
  failover_hist["${failover_iter}"]=$(( ${failover_hist["${failover_iter}"]:-0} + 1 ))

  echo "[stress] run ${i}/${rounds}: rc=${rc} result=${result} failover_iter=${failover_iter}" | tee -a "${SUMMARY_LOG}"
done

{
  echo
  echo "===== stress summary ====="
  echo "rounds=${rounds}"
  echo "pass=${pass_count}"
  echo "diff=${diff_count}"
  echo "segment_fault=${segfault_count}"
  echo "timeout=${timeout_count}"
  echo "unknown=${unknown_count}"
  echo "summary_log=${SUMMARY_LOG}"
  echo "log_dir=${LOG_DIR}"
} | tee -a "${SUMMARY_LOG}"
