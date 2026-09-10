#!/usr/bin/env bash
# Shared settings and helpers for the cloudlab_r7525 tests. Source this file; do not execute it.
#
# Testbed: node-1 (run everything from here) + node-2, 2x V100S each. NCCL HCAs: mlx5_0 (25G, also the
# bootstrap interface eno33np0), mlx5_2 and mlx5_3 (100G). The "failed" NIC in every scenario is node-1's
# mlx5_2 (NCCL netDev 1): the hot-repair tests really cut it on the SmartNIC (nic/disconnect_nic1.sh), the
# nccl-tests scenarios only declare it failed (R2CC_FAILED_HCA).
#
# Variables a script may export BEFORE sourcing this file:
#   NCCL_IB_HCA_LIST   HCAs NCCL may use (default all three)
#   R2CC_FAILED_NODE / R2CC_FAILED_HCA   failure model for the R2CC scenarios (default node 0 / mlx5_2)

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EXAMPLE_DIR}/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-node-2}"
LOG_DIR="${LOG_DIR:-${EXAMPLE_DIR}/logs/local}"   # only used when SAVE_LOG=1 (logs/ itself holds the reference logs)
NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/mydata/nccl-tests}"
NCCL_IB_HCA_LIST="${NCCL_IB_HCA_LIST:-mlx5_0,mlx5_2,mlx5_3}"

# Failure model (identical on all ranks; the library cross-checks these across ranks and adopts rank 0's).
export R2CC_FAILED_NODE="${R2CC_FAILED_NODE:-0}"     # node index of the degraded server (node-1)
export R2CC_FAILED_HCA="${R2CC_FAILED_HCA:-mlx5_2}"   # HCA name(s) or netDev index(es) assumed failed there

# OpenMPI must not try to reach peers through the SmartNIC management network (192.168.100.x).
export OMPI_MCA_btl_tcp_if_include="${OMPI_MCA_btl_tcp_if_include:-eno33np0}"

MPIRUN_BASE=(
  mpirun -np 4 -host "localhost:2,${REMOTE_HOST}:2"
  -mca pml ob1 -mca btl tcp,self -mca btl_tcp_if_include eno33np0
  -x NCCL_NET_GDR_LEVEL=SYS -x NCCL_IB_GID_INDEX=3 -x "NCCL_TOPO_FILE=${HOME}/topo.xml"
  -x NCCL_SOCKET_IFNAME=eno33np0 -x "NCCL_IB_HCA=${NCCL_IB_HCA_LIST}" -x NCCL_IB_MERGE_NICS=0
  -x NCCL_ALGO=Ring
)

# Tuning variables forwarded to all ranks when the caller has set them
# (OpenMPI aborts on `-x VAR` if VAR is unset, so only set ones are forwarded).
R2CC_PASSTHROUGH_VARS=(
  R2CC_FAILOVER_TIMEOUT_MS NCCL_R2CC_FAILOVER_TIMEOUT_MS
  R2CC_MODE R2CC_FAILED_NODE R2CC_FAILED_HCA R2CC_FAILED_NIC_COUNT
  R2CC_AR_STAGE2_CHUNKS R2CC_AR_SCHEDULE R2CC_AR_MIN_BYTES R2CC_AR_AFTER_REPAIR R2CC_USE_ALL_NIC
  NCCL_DEBUG_SUBSYS NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS
)

# Refuse to start while another multi-node job is running (one job at a time on this testbed).
check_idle() {
  local busy
  # Match the launched binaries / launchers themselves, not shells whose command line merely mentions them.
  local pat='hot_repair/test_hot_repair( |$)|nccl-tests/build/[a-z_]+_perf( |$)|^[0-9]+ mpirun -np|^[0-9]+ orted -mca'
  busy="$( { pgrep -af "${pat}" ;
             ssh -o BatchMode=yes -o ConnectTimeout=5 "${REMOTE_HOST}" "pgrep -af '${pat}'" ; } 2>/dev/null \
           | grep -v -e pgrep || true)"
  if [[ -n "${busy}" ]]; then
    echo "Another multi-node job is running; run tools/kill.sh first:" >&2
    echo "${busy}" >&2
    return 1
  fi
}

# Make sure node-1's mlx5_2 is not still blocked on the SmartNIC (e.g. a hot-repair run that was killed while
# disconnected). connect_nic1.sh only deletes the drop rule by cookie, so it is safe to call every time.
# The hot-repair driver does the same in its own preflight (plus a ping check); the nccl-tests scripts call it here.
ensure_nic_connected() {
  if ! "${EXAMPLE_DIR}/nic/connect_nic1.sh"; then
    echo "WARNING: nic/connect_nic1.sh failed (is 'ssh nic' configured?); continuing" >&2
  fi
}

# Default nccl-tests arguments: full size sweep, results verified against a CPU reference (-c 1).
NCCL_TESTS_DEFAULT_ARGS=(-b 8 -e 4G -f 2 -g 1 -c 1 -n 5 -w 2 -d float -o sum)

# run_nccl_tests <R2CC_MODE> <log_tag> [nccl-tests args...]
# Runs $NCCL_TEST_BIN (default all_reduce_perf) from $NCCL_TESTS_DIR/build on the 4 GPUs. The "#wrong"
# columns and the final "Out of bounds values : 0 OK" line must be 0 when -c 1 is used.
run_nccl_tests() {
  local mode="$1" tag="$2"; shift 2
  local bin="${NCCL_TESTS_DIR}/build/${NCCL_TEST_BIN:-all_reduce_perf}"
  if [[ ! -x "${bin}" ]]; then
    echo "nccl-tests binary not found: ${bin}" >&2
    echo "Build it (and copy it to ${REMOTE_HOST}) with: ${EXAMPLE_DIR}/tools/build_nccl_tests.sh" >&2
    return 1
  fi
  local -a args=("$@")
  [[ ${#args[@]} -eq 0 ]] && args=("${NCCL_TESTS_DEFAULT_ARGS[@]}")
  export R2CC_MODE="${mode}"
  local -a envargs=() v
  for v in "${R2CC_PASSTHROUGH_VARS[@]}"; do [[ -v "$v" ]] && envargs+=(-x "$v"); done
  # Output goes to stdout only; the calling test script captures its whole terminal output into logs/<NN>.<name>.log.
  # A copy of this scenario's output is kept in a temporary file for result parsing (busbw_table).
  local tmp; tmp="$(mktemp)"
  echo "[nccl-tests] R2CC_MODE=${mode} NCCL_IB_HCA=${NCCL_IB_HCA_LIST} R2CC_FAILED_NODE=${R2CC_FAILED_NODE} R2CC_FAILED_HCA=${R2CC_FAILED_HCA}"
  echo "[nccl-tests] ${bin##*/} ${args[*]}"
  set +e
  timeout "${NCCL_TESTS_TIMEOUT:-900}" "${MPIRUN_BASE[@]}" "${envargs[@]}" -x "NCCL_DEBUG=${NCCL_DEBUG:-WARN}" \
    "${bin}" "${args[@]}" 2>&1 | tee "${tmp}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "[nccl-tests] rc=${rc} (124 = timeout) - cleaning up leftover processes" >&2
    "${EXAMPLE_DIR}/tools/kill.sh" >/dev/null 2>&1 || true
  else
    echo "[nccl-tests] rc=0  $(grep -oE 'Out of bounds values : [0-9]+ [A-Z]+' "${tmp}" | tail -1)"
  fi
  if [[ -n "${NCCL_TESTS_RESULT_FILE:-}" ]]; then mv "${tmp}" "${NCCL_TESTS_RESULT_FILE}"; else rm -f "${tmp}"; fi
  return "${rc}"
}

# run_logged <NN.name> <command...>: run the command. By default nothing is written to disk: the six files in
# logs/ are the reference logs shipped with the repository and must not be overwritten by a local run. With
# SAVE_LOG=1 the complete terminal output is also written to ${LOG_DIR}/<NN.name>.log (default logs/local/,
# git-ignored). To regenerate the reference logs deliberately: SAVE_LOG=1 LOG_DIR=<repo>/examples/cloudlab_r7525/logs.
run_logged() {
  local name="$1"; shift
  if [[ "${SAVE_LOG:-0}" != "1" ]]; then "$@"; return; fi
  mkdir -p "${LOG_DIR}"
  echo "[log] saving terminal output to ${LOG_DIR}/${name}.log"
  set +e
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  return "${rc}"
}

# busbw_table <label:log> [<label:log> ...]
# Prints one row per message size: out-of-place/in-place bus bandwidth (GB/s) and #wrong for each log.
busbw_table() {
  local spec labels=() logs=()
  for spec in "$@"; do labels+=("${spec%%:*}"); logs+=("${spec#*:}"); done
  awk -v labels="$(IFS=,; echo "${labels[*]}")" '
    BEGIN { nl = split(labels, L, ","); }
    FNR == 1 { fi++; }
    /^ +[0-9]+ +[0-9]+ +[a-z0-9]+ +[a-z]+ +-?[0-9]+ / {
      size = $1; if (!(size in seen)) { seen[size] = 1; order[++n] = size; }
      cell[size, fi] = sprintf("%s/%s (wrong %s/%s)", $8, $12, $9, $13);
    }
    END {
      printf "%-12s", "bytes"; for (i = 1; i <= nl; i++) printf " | %-32s", L[i]; printf "\n";
      for (k = 1; k <= n; k++) {
        size = order[k]; printf "%-12s", size;
        for (i = 1; i <= nl; i++) printf " | %-32s", ((size, i) in cell ? cell[size, i] : "-");
        printf "\n";
      }
      print "(cells: busbw out-of-place/in-place GB/s, then #wrong out-of-place/in-place)";
    }' "${logs[@]}"
}
