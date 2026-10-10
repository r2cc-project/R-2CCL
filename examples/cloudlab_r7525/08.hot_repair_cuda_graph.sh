#!/usr/bin/env bash
# The hot repair of 01/02 with the AllReduce captured in a CUDA graph (README.md, section 4). Two runs of
# hot_repair/test_hot_repair_graph, each 10 x 4 GiB with node-1's mlx5_2 cut 4 s after the start, i.e. during
# iteration 3. The graph captured at the start is replayed in every iteration, so the hot repair happens while it is
# replayed; after iteration 6 the AllReduce is captured again:
#   run 1   as R2CC-Balance (R2CC_AR_AFTER_REPAIR=2)
#   run 2   as R2CC-AllReduce (R2CC_AR_AFTER_REPAIR=3); one AllReduce outside the capture first creates its
#           sub-communicators
#   Every iteration gets new inputs, and every element of every AllReduce output is checked on every rank, as in
#   01/02. The script prints one table: the time per iteration of the graph captured before the failure (healthy,
#   then after the hot repair) and of the graph captured again, and the re-capture overhead: the time of the
#   iteration that performs the re-capture, including the eager collective that R2CC-AllReduce needs to create its
#   sub-communicators, minus the time of an iteration replayed from the new graph. Below it, the result of the check
#   and the evidence that the cut hit a running AllReduce, for each run. The exit code is that of 01/02 (section 2):
#   0 when both runs pass. A run that fails, or does not finish within RUN_TIMEOUT seconds (default 120), is
#   stopped, mlx5_2 is restored, and its complete output is kept (the path is printed). About 1.5 minutes.
#
# Usage: ./08.hot_repair_cuda_graph.sh
#   R2CC_TEST_CORRUPT=<n>[,nan] tests the check, as in 01/02 (README.md, section 3.1).
#   Nothing is written to disk by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${EXAMPLE_DIR}"
check_idle
ensure_nic_connected

# Stops what a timed-out run left behind on every node and restores mlx5_2.
stop_run() {
  local pat='/[t]est_hot_repair_graph|[m]pirun -np|[o]rted -mca' h
  pkill -9 -f "${pat}" || true
  for h in "${REMOTE_HOST_LIST[@]}"; do ssh -o BatchMode=yes "${h}" "pkill -9 -f '${pat}' || true" || true; done
  ./nic/connect_nic1.sh > /dev/null 2>&1 || true
}

main() {
  local rc=0 out rcs="" nrun=0 h ip1 ip2 ip3
  out="$(mktemp -d)"
  print_nic_limit
  for h in "${REMOTE_HOST_LIST[@]}"; do   # warm the ARP entries of the SmartNIC ports of the other servers
    read -r ip1 ip2 ip3 <<< "$(node_ips "${h}")"
    ping -c 1 -W 1 "${ip2}" > /dev/null 2>&1 || true
    ping -c 1 -W 1 "${ip3}" > /dev/null 2>&1 || true
  done

  run() {  # run <label> <R2CC_AR_AFTER_REPAIR> [test_hot_repair_graph args]; the output goes to ${out}/<run number>
    local label="$1" after="$2" r; shift 2
    local -a test=()
    [[ -v R2CC_TEST_CORRUPT ]] && test=(-x R2CC_TEST_CORRUPT)
    nrun=$((nrun + 1))
    echo "[08] ${label}"
    set +e
    timeout -k 10 "${RUN_TIMEOUT:-120}" "${MPIRUN_BASE[@]}" -x "NCCL_DEBUG=${NCCL_DEBUG:-WARN}" \
      -x NCCL_R2CC_FAILOVER_TIMEOUT_MS=5000 -x NCCL_IB_TIMEOUT=16 -x NCCL_IB_RETRY_CNT=1 \
      -x R2CC_FAILED_NODE -x R2CC_FAILED_HCA -x "R2CC_AR_AFTER_REPAIR=${after}" "${test[@]}" \
      ./hot_repair/test_hot_repair_graph "$@" > "${out}/${nrun}" 2>&1
    r=$?
    set -e
    if [[ ${r} -ne 0 && ${r} -ne 2 && ${r} -ne 3 ]]; then stop_run; fi
    rcs+="${rcs:+,}${r}"
    # 2 (wrong results) outranks an abort, which outranks 3 (the cut did not hit a running AllReduce)
    if [[ ${r} -eq 2 ]]; then rc=2
    elif [[ ${r} -ne 0 && ${r} -ne 3 && ${rc} -ne 2 ]]; then rc=1
    elif [[ ${r} -eq 3 && ${rc} -eq 0 ]]; then rc=3; fi
  }
  run "run 1/2: re-captured as R2CC-Balance after iteration 6" 2 --recapture-at 6
  run "run 2/2: re-captured as R2CC-AllReduce after iteration 6" 3 --recapture-at 6 --eager

  echo
  awk -v rcs="${rcs}" -v limit="${RUN_TIMEOUT:-120}" '
    function mean(f, lo, hi,   i, s, k) {
      s = 0; k = 0
      for (i = lo; i <= hi; i++) if ((f, i) in t) { s += t[f, i]; k++ }
      return k ? s / k : -1
    }
    function sec(f, lo, hi,   m) { m = mean(f, lo, hi); return m < 0 ? "-" : sprintf("%.2f s", m / 1000) }
    function sec2(lo1, hi1, lo2, hi2,   i, s, k) {   # mean over the given iterations of both runs
      s = 0; k = 0
      for (i = lo1; i <= hi1; i++) if ((1, i) in t) { s += t[1, i]; k++ }
      for (i = lo2; i <= hi2; i++) if ((2, i) in t) { s += t[2, i]; k++ }
      return k ? sprintf("%.2f s", s / k / 1000) : "-"
    }
    function failed(f) {
      return RC[f] == 124 || RC[f] == 137 ? "not finished within " limit " s; stopped, mlx5_2 restored" : \
             RC[f] == 2 ? "wrong results (exit 2)" : \
             RC[f] == 3 ? "the cut did not hit a running AllReduce (exit 3); run again" : \
             RC[f] != 0 ? "failed (exit " RC[f] ")" : "mlx5_2 was not cut during the run"
    }
    function overhead(f,   o) {   # re-capture iteration (eager collective or first replay, plus capture) minus a replay
      o = capms[f] + instms[f] + (eagerms[f] != "" ? eagerms[f] - mean(f, re[f] + 1, n[f]) : 0)
      return o < 100 ? sprintf("%.1f ms", o) : sprintf("%.2f s", o / 1000)
    }
    BEGIN { split(rcs, RC, ","); for (k = 1; k < ARGC; k++) idx[ARGV[k]] = k }
    { f = idx[FILENAME] }
    /^Config: / { if (match($0, /recapture_at=-?[0-9]+/)) re[f] = substr($0, RSTART + 13, RLENGTH - 13) + 0 }
    /^\[Rank 0\] Captured AllReduce/ && ++nc[f] == 2 {
      if (match($0, /capture [0-9.]+ ms/)) capms[f] = substr($0, RSTART + 8, RLENGTH - 11) + 0
      if (match($0, /instantiate [0-9.]+ ms/)) instms[f] = substr($0, RSTART + 12, RLENGTH - 15) + 0
    }
    /^\[Rank 0\] Eager AllReduce/ { if (match($0, /took [0-9]+ ms/)) eagerms[f] = substr($0, RSTART + 5, RLENGTH - 8) + 0 }
    /^\[Rank 0\] (Verification|Failure evidence|TEST PASS|TEST FAIL)/ { sub(/^\[Rank 0\] /, ""); check[f] = check[f] "  " $0 "\n" }
    /^Iter +Time\(ms\)/ { tab[f] = 1; next }
    tab[f] && /^[0-9]+ +[0-9]+ / { i = $1 + 0; t[f, i] = $2; r2[f, i] = $4; n[f] = i; next }
    { tab[f] = 0 }
    END {
      for (f = 1; f <= 2; f++) {    # the iteration the cut hits: the first in which mlx5_2 received under 90% of its share
        full = 0; for (i = 1; i <= n[f]; i++) if (r2[f, i] > full) full = r2[f, i]
        for (i = 1; i <= n[f] && !cut[f]; i++) if (r2[f, i] < 0.9 * full) cut[f] = i
        ok[f] = RC[f] == 0 && cut[f] > 0
      }
      print "===== CUDA Graphs: 4 GiB AllReduce on all GPUs, mlx5_2 of node-1 cut during the run ====="
      fmt = "%-34s %12s   %s\n"
      printf fmt, "Replayed graph (4 GiB AllReduce)", "Time / iter", "Re-capture overhead*"
      if (ok[1] && ok[2]) {     # both runs replay the graph captured before the failure until iteration 6
        printf fmt, "Pre-failure graph, healthy", sec2(2, cut[1] - 1, 2, cut[2] - 1), "--"
        printf fmt, "Pre-failure graph, HotRepair", sec2(cut[1] + 1, re[1], cut[2] + 1, re[2]), "--"
      } else printf "%-34s %s\n", "Pre-failure graph", ok[1] ? failed(2) : failed(1)
      if (ok[1]) printf fmt, "Re-captured as R2CC-Balance", sec(1, re[1] + 1, n[1]), overhead(1)
      else printf "%-34s %s\n", "Re-captured as R2CC-Balance", failed(1)
      if (ok[2]) printf fmt, "Re-captured as R2CC-AllReduce", sec(2, re[2] + 1, n[2]), overhead(2)
      else printf "%-34s %s\n", "Re-captured as R2CC-AllReduce", failed(2)
      print "* the time of the iteration that performs the re-capture, including the eager collective that R2CC-AllReduce"
      print "  needs to create its sub-communicators, minus the time of an iteration replayed from the new graph"
      print ""
      label[1] = "run 1/2, re-captured as R2CC-Balance:"; label[2] = "run 2/2, re-captured as R2CC-AllReduce:"
      for (f = 1; f <= 2; f++) printf "%s\n%s", label[f], (check[f] != "" ? check[f] : "  no result (exit " RC[f] ")\n")
    }' "${out}/1" "${out}/2"
  case "${rc}" in
    0) echo "[result] exit=0 PASS" ;;
    2) echo "[result] exit=2 FAIL: wrong results" ;;
    3) echo "[result] exit=3 FAIL: the NIC failure did not hit a running AllReduce (see 'Failure evidence'); run again" ;;
    *) echo "[result] exit=${rc} FAIL: aborted (CUDA/NCCL/MPI error, crash, kill or timeout)" ;;
  esac
  if [[ ${rc} -eq 0 ]]; then rm -rf "${out}"; else echo "complete output of the runs: ${out}"; fi
  return "${rc}"
}
run_logged 08.hot_repair_cuda_graph main
