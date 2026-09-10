#!/usr/bin/env bash
set -euo pipefail

# Test drivers, the hot-repair binary, nccl-tests binaries and the mpirun/orted that launched them.
PATTERN='[0-9][0-9]\.[a-z_]*\.sh|[s]tress_test\.sh|[r]un_hot_repair\.sh|[h]ot_repair_to_[a-z_]*\.sh|[n]ccl_tests_[a-z_]*\.sh|/[t]est_hot_repair|[a-z_]*_perf( |$)|[m]pirun -np|[o]rted -mca'
REMOTE_HOST="${REMOTE_HOST:-node-2}"
LOCAL_ONLY=0

if [[ "${1:-}" == "--local-only" ]]; then
  LOCAL_ONLY=1
fi

kill_target_on_host() {
  local host="$1"
  local prefix="$2"

  echo "[kill] ${prefix}: stopping processes matching ${PATTERN}"
  if [[ "${host}" == "local" ]]; then
    pkill -f "${PATTERN}" || true
    sleep 1
    pkill -9 -f "${PATTERN}" || true
    pgrep -af "${PATTERN}" || true
  else
    ssh -o BatchMode=yes -o ConnectTimeout=3 "${host}" \
      "pkill -f '${PATTERN}' || true; sleep 1; pkill -9 -f '${PATTERN}' || true; pgrep -af '${PATTERN}' || true" \
      || echo "[kill] ${prefix}: skip (ssh failed)"
  fi
}

kill_target_on_host "local" "local"

if [[ "${LOCAL_ONLY}" -eq 0 ]]; then
  kill_target_on_host "${REMOTE_HOST}" "remote(${REMOTE_HOST})"
fi

echo "[kill] done"
