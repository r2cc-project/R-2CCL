#!/usr/bin/env bash
set -euo pipefail

# CLI:
#   ./run_hot_repair.sh
#   ./run_hot_repair.sh -log 1   # enable NCCL INFO trace logs
#   ./run_hot_repair.sh -log 0   # disable NCCL INFO trace logs (default)
LOG_FLAG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -log)
      if [[ $# -lt 2 ]]; then
        echo "Usage: $0 [-log 0|1]"
        exit 1
      fi
      if [[ "$2" != "0" && "$2" != "1" ]]; then
        echo "Invalid value for -log: $2 (expected 0 or 1)"
        echo "Usage: $0 [-log 0|1]"
        exit 1
      fi
      LOG_FLAG="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-log 0|1]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [-log 0|1]"
      exit 1
      ;;
  esac
done

# Channel-level proxy-state tracing (used by sendProxyProgress/recvProxyProgress).
# Usage examples:
#   ./run_hot_repair.sh                  # all channels
#   TRACE_CH=3 ./run_hot_repair.sh       # single channel
#   TRACE_CH=8 TRACE_STALL_ITERS=5000 ./run_hot_repair.sh
TRACE_CH="${TRACE_CH:--1}"
TRACE_STALL_ITERS="${TRACE_STALL_ITERS:-2000}"
TRACE_TRANSITIONS="${TRACE_TRANSITIONS:-0}"
if [[ "${LOG_FLAG}" == "1" ]]; then
  TRACE_ONLY="${TRACE_ONLY:-1}"
else
  TRACE_ONLY="${TRACE_ONLY:-0}"
fi
TRACE_GREP="${TRACE_GREP:-R2CC_PROXY_STATE|\\[trace\\]|\\[Rank [0-9]+\\] Running on|\\[Rank 0\\] Iter [0-9]+/[0-9]+|Config:|IB RX per-iteration|TEST PASS|TEST FAIL|NCCL WARN|NCCL ERROR|Failed}"
SAVE_RAW_LOG="${SAVE_RAW_LOG:-0}"
LOG_DIR="${LOG_DIR:-./logs}"
LOG_TAG="${LOG_TAG:-$(date +%Y%m%d_%H%M%S)}"
SAMPLE_LINES="${SAMPLE_LINES:-0}"

export R2CC_TRACE_CHANNEL="${TRACE_CH}"
export R2CC_TRACE_STALL_ITERS="${TRACE_STALL_ITERS}"
export R2CC_TRACE_TRANSITIONS="${TRACE_TRANSITIONS}"
if [[ "${LOG_FLAG}" == "1" ]]; then
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-R2CC}"
else
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-ALL}"
fi

mkdir -p "${LOG_DIR}"
RAW_LOG="${LOG_DIR}/test_hot_repair_${LOG_TAG}.raw.log"
TRACE_LOG="${LOG_DIR}/test_hot_repair_${LOG_TAG}.trace.log"

preflight_ping_target() {
  local ip="$1"
  local tries="${2:-5}"
  local count="${3:-1}"
  local wait_s="${4:-1}"
  local i
  for ((i=1; i<=tries; i++)); do
    if ping -c "${count}" -W "${wait_s}" "${ip}" >/dev/null 2>&1; then
      echo "[preflight] ping ${ip} OK (attempt ${i}/${tries})"
      return 0
    fi
    echo "[preflight] ping ${ip} failed (attempt ${i}/${tries})"
    sleep 1
  done
  return 1
}

run_preflight_connect() {
  local connect_cmd="./nic/connect_nic1.sh"
  local settle_s="${PREFLIGHT_CONNECT_SETTLE_S:-1}"

  echo "[preflight] running connect cmd: ${connect_cmd}"
  if ! "${connect_cmd}"; then
    echo "[preflight] ERROR: connect cmd failed: ${connect_cmd}"
    return 1
  fi

  if [[ "${settle_s}" =~ ^[0-9]+$ ]] && (( settle_s > 0 )); then
    sleep "${settle_s}"
  fi
  echo "[preflight] connect done"
}

run_preflight_ping() {
  local ping_tries="${PREFLIGHT_PING_TRIES:-8}"
  local ping_count="${PREFLIGHT_PING_COUNT:-1}"
  local ping_wait="${PREFLIGHT_PING_WAIT_S:-1}"
  local target_ips=("10.10.2.5" "10.10.3.6")

  echo "[preflight] start ping warmup targets=${target_ips[*]} tries=${ping_tries} count=${ping_count} wait_s=${ping_wait}"
  local ip
  for ip in "${target_ips[@]}"; do
    if ! preflight_ping_target "${ip}" "${ping_tries}" "${ping_count}" "${ping_wait}"; then
      echo "[preflight] ERROR: ${ip} unreachable before test; abort."
      return 1
    fi
    # Extra packet to warm ARP/flow cache right before test starts.
    ping -c 1 -W "${ping_wait}" "${ip}" >/dev/null 2>&1 || true
  done
  echo "[preflight] ping warmup done"
}

extract_failover_iter() {
  local run_log="$1"
  awk '
    BEGIN {
      in_table = 0;
      col = -1;
      found = 0;
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
          found = 1;
          exit;
        }
      }
      next;
    }
    END {
      if (!found) print "NA";
    }
  ' "${run_log}"
}

echo "[trace] log=${LOG_FLAG} channel=${R2CC_TRACE_CHANNEL} stall_iters=${R2CC_TRACE_STALL_ITERS} transitions=${R2CC_TRACE_TRANSITIONS} NCCL_DEBUG=${NCCL_DEBUG} TRACE_ONLY=${TRACE_ONLY}"
echo "[trace] trace_log=${TRACE_LOG} save_raw_log=${SAVE_RAW_LOG}"

run_preflight_connect
run_preflight_ping

mpirun_cmd=(
  mpirun -np 4
  -host localhost:2,node-2:2
  -mca pml ob1 -mca btl tcp,self
  -x R2CC_TRACE_CHANNEL
  -x R2CC_TRACE_STALL_ITERS
  -x R2CC_TRACE_TRANSITIONS
  -x NCCL_DEBUG
  -x NCCL_DEBUG_SUBSYS
  -x NCCL_NET_GDR_LEVEL=SYS
  -x NCCL_IB_GID_INDEX=3
  -x NCCL_TOPO_FILE=~/topo.xml
  -x NCCL_SOCKET_IFNAME=eno33np0
  -x NCCL_IB_HCA=mlx5_0,mlx5_2,mlx5_3
  -x NCCL_IB_MERGE_NICS=0
  -x NCCL_R2CC_FAILOVER_TIMEOUT_MS=5000
  -x NCCL_IB_TIMEOUT=16
  -x NCCL_IB_RETRY_CNT=1
  -x NCCL_ALGO=Ring
  -x R2CC_AR_START_DISCONNECT_DELAY_MS=4000
  ./test_hot_repair
)

set +e
if [[ "${SAVE_RAW_LOG}" == "1" ]]; then
  stdbuf -oL -eL "${mpirun_cmd[@]}" > "${RAW_LOG}" 2>&1
  mpirun_rc=$?
  if [[ "${TRACE_ONLY}" == "1" ]]; then
    grep -E "${TRACE_GREP}" "${RAW_LOG}" > "${TRACE_LOG}" || true
  else
    cp "${RAW_LOG}" "${TRACE_LOG}"
  fi
else
  if [[ "${TRACE_ONLY}" == "1" ]]; then
    stdbuf -oL -eL "${mpirun_cmd[@]}" 2>&1 \
      | grep --line-buffered -E "${TRACE_GREP}" \
      | tee "${TRACE_LOG}"
    mpirun_rc=${PIPESTATUS[0]}
  else
    stdbuf -oL -eL "${mpirun_cmd[@]}" 2>&1 | tee "${TRACE_LOG}"
    mpirun_rc=$?
  fi
fi
set -e

VIEW_LOG="${TRACE_LOG}"

view_lines="$(wc -l < "${VIEW_LOG}")"
state_lines="$(grep -c 'R2CC_PROXY_STATE' "${VIEW_LOG}" || true)"
failover_iter="$(extract_failover_iter "${VIEW_LOG}")"

if [[ "${SAVE_RAW_LOG}" == "1" ]]; then
  raw_lines="$(wc -l < "${RAW_LOG}")"
  echo "[trace] raw_log=${RAW_LOG} raw_lines=${raw_lines}"
fi
echo "[trace] view_lines=${view_lines} state_lines=${state_lines}"
echo "[trace] view_log=${VIEW_LOG}"
echo "[trace] failover_iter=${failover_iter} rule=first iteration where mlx5_2_RX < 20"
if [[ "${SAMPLE_LINES}" =~ ^[0-9]+$ ]] && (( SAMPLE_LINES > 0 )); then
  echo "----- SAMPLE HEAD (${SAMPLE_LINES}) -----"
  head -n "${SAMPLE_LINES}" "${VIEW_LOG}" || true
  echo "----- SAMPLE TAIL (${SAMPLE_LINES}) -----"
  tail -n "${SAMPLE_LINES}" "${VIEW_LOG}" || true
fi

exit "${mpirun_rc}"

# NCCL_R2CC_FAILOVER_TIMEOUT_MS=500   Should set according your scale and NCCL native timeout and retry env.
# NCCL_IB_TIMEOUT=16   Default 20 ~ 4s timeout. 16 ~ 0.2s.
# NCCL_IB_RETRY_CNT=3  Default 7 times retry.
