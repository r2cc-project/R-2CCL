#!/usr/bin/env bash
# Repeat one of the hot-repair tests N times and summarise pass/diff/segfault/timeout counts and the
# iteration in which failover happened (first iteration with mlx5_2 RX < 20 MB).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: tools/stress_test.sh -n <rounds> [-t <test script>]

Options:
  -n <rounds>        Number of runs.
  -t <test script>   Test to repeat (default: 01.hot_repair_to_balance.sh; e.g. 02.hot_repair_to_r2cc_allreduce.sh).
  -h, --help         Show this help.
EOF
}

extract_failover_iter() {
  local run_log="$1"
  awk '
    BEGIN { in_table = 0; col = -1; }
    /^Iter[[:space:]]+Time\(ms\)/ {
      in_table = 1; col = -1;
      for (i = 1; i <= NF; i++) if ($i == "mlx5_2_RX") { col = i; break; }
      next;
    }
    in_table {
      if (NF == 0 || $1 !~ /^[0-9]+$/) { in_table = 0; next; }
      if (col > 0 && col <= NF && $(col) + 0 < 20) { print $1; exit; }
      next;
    }
  ' "${run_log}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TEST_SCRIPT="01.hot_repair_to_balance.sh"
LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/logs/stress}"; export LOG_DIR   # wrapper logs of stress rounds stay out of logs/<NN>.<name>.log
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

rounds=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n) [[ $# -ge 2 ]] || { usage; exit 1; }; rounds="$2"; shift 2 ;;
    -t) [[ $# -ge 2 ]] || { usage; exit 1; }; TEST_SCRIPT="$(basename "$2")"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done
if [[ -z "${rounds}" || ! "${rounds}" =~ ^[0-9]+$ || "${rounds}" -le 0 ]]; then
  echo "Invalid -n value: ${rounds:-<empty>}"; usage; exit 1
fi
TEST_CMD="${EXAMPLE_DIR}/${TEST_SCRIPT}"
if [[ ! -x "${TEST_CMD}" ]]; then
  echo "test script not found or not executable: ${TEST_CMD}"; exit 1
fi

mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/stress_summary_${RUN_TAG}.log"

pass_count=0; diff_count=0; segfault_count=0; timeout_count=0; unknown_count=0
declare -A failover_hist=()

for ((i=1; i<=rounds; i++)); do
  run_log="${LOG_DIR}/run_${RUN_TAG}_$(printf '%03d' "${i}").log"
  echo "[stress] run ${i}/${rounds}: ${TEST_SCRIPT} timeout=150s log=${run_log}" | tee -a "${SUMMARY_LOG}"

  set +e
  ( cd "${EXAMPLE_DIR}" && LOG_TAG="stress_${RUN_TAG}_${i}" timeout 150 "${TEST_CMD}" ) > "${run_log}" 2>&1
  rc=$?
  set -e
  "${SCRIPT_DIR}/kill.sh" >/dev/null 2>&1 || true

  result="unknown"
  if grep -qiE 'Segmentation fault|Caught signal 11|exited on signal 11|Signal: Segmentation fault' "${run_log}"; then
    result="segment_fault"; ((segfault_count+=1))
  elif grep -q 'diff=' "${run_log}" || grep -q 'TEST FAIL: Verification failed' "${run_log}"; then
    result="diff"; ((diff_count+=1))
  elif grep -q 'TEST PASS' "${run_log}"; then
    result="pass"; ((pass_count+=1))
  elif [[ "${rc}" -eq 124 ]]; then
    result="timeout"; ((timeout_count+=1))
  else
    ((unknown_count+=1))
  fi

  failover_iter="$(extract_failover_iter "${run_log}")"
  [[ -z "${failover_iter}" ]] && failover_iter="NA"
  failover_hist["${failover_iter}"]=$(( ${failover_hist["${failover_iter}"]:-0} + 1 ))

  echo "[stress] run ${i}/${rounds}: rc=${rc} result=${result} failover_iter=${failover_iter}" | tee -a "${SUMMARY_LOG}"
done

{
  echo
  echo "===== stress summary ====="
  echo "test=${TEST_SCRIPT}"
  echo "rounds=${rounds}"
  echo "pass=${pass_count}"
  echo "diff=${diff_count}"
  echo "segment_fault=${segfault_count}"
  echo "timeout=${timeout_count}"
  echo "unknown=${unknown_count}"
  echo -n "failover_iter histogram:"; for k in "${!failover_hist[@]}"; do echo -n " ${k}x${failover_hist[$k]}"; done; echo
  echo "summary_log=${SUMMARY_LOG}"
  echo "log_dir=${LOG_DIR}"
} | tee -a "${SUMMARY_LOG}"
