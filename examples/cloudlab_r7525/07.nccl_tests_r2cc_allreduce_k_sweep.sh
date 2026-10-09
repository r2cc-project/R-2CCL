#!/usr/bin/env bash
# R2CC-AllReduce with K = 1, 2, 4, 8, 16 Stage-2 chunks, against the model of README.md section 1.3.
#
#   nccl-tests at 4 GiB on the degraded cluster of 05/06 (node-1's mlx5_2 declared failed, X = 1/3): healthy and
#   R2CC-Balance, then R2CC-AllReduce for every K (R2CC_AR_STAGE2_CHUNKS) with Stage 1 concurrent
#   (R2CC_AR_SCHEDULE=1, the default with three or more servers), then healthy and Balance again. The table at the
#   end gives the time relative to healthy, separately for out-of-place and in-place (T0 = mean of the two healthy
#   runs), next to the model:
#     R2CC-Balance     T/T0 = 1/(1-X)
#     R2CC-AllReduce   T/T0 = 1 + X/((1-X) * 2(P-1)/P) * (1 + 1/K)        P = number of ranks
#   -c 1 checks every element of one extra AllReduce of each run, as in 03-06. About 9 minutes.
#
# Usage: ./07.nccl_tests_r2cc_allreduce_k_sweep.sh [nccl-tests args...]
#   default args: -b 4G -e 4G -f 2 -g 1 -c 1 -n 5 -w 2 -d float -o sum   (the table uses the largest size)
#   K_LIST="1 2 4 8 16" (env) selects the K values.
#   Nothing is written to disk by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${EXAMPLE_DIR}/common.sh"
check_idle
ensure_nic_connected

main() {
  local ARGS=("$@") runs=() K
  [[ ${#ARGS[@]} -eq 0 ]] && ARGS=(-b 4G -e 4G -f 2 -g 1 -c 1 -n 5 -w 2 -d float -o sum)
  TMP="$(mktemp -d)"; trap 'rm -rf "${TMP}"' EXIT

  run() {  # run <tag> <R2CC_MODE> [K]; each run in a subshell so its settings cannot leak into the next one
    local tag="$1" mode="$2" k="${3:-}"
    echo; echo "===== ${tag} ====="
    ( source "${EXAMPLE_DIR}/common.sh"
      if [[ -n "${k}" ]]; then
        export R2CC_AR_STAGE2_CHUNKS="${k}" R2CC_AR_SCHEDULE=1
        echo "[k-sweep] R2CC_AR_STAGE2_CHUNKS=${k} R2CC_AR_SCHEDULE=1"
      fi
      NCCL_TESTS_RESULT_FILE="${TMP}/${tag}" run_nccl_tests "${mode}" "${tag}" "${ARGS[@]}" )
    runs+=("${tag}:${k:--}")
  }
  run healthy_1 0
  run balance_1 2
  for K in ${K_LIST:-1 2 4 8 16}; do run "r2cc_allreduce_K${K}" 3 "${K}"; done
  run healthy_2 0
  run balance_2 2

  local nnic nfail files=() spec
  nnic="$(tr ',' '\n' <<< "${NCCL_IB_HCA_LIST}" | grep -c .)"
  nfail="$(tr ',' '\n' <<< "${R2CC_FAILED_HCA}" | grep -c .)"
  for spec in "${runs[@]}"; do files+=("${TMP}/${spec%%:*}"); done
  echo
  echo "===== time relative to healthy vs. the model (${ARGS[*]}; X = ${nfail}/${nnic}, P = ${NRANKS}) ====="
  awk -v specs="$(IFS=,; echo "${runs[*]}")" -v X="$(awk -v a="${nfail}" -v b="${nnic}" 'BEGIN { print a / b }')" \
      -v P="${NRANKS}" '
    BEGIN { n = split(specs, S, ","); }
    FNR == 1 { fi++; }
    /^ +[0-9]+ +[0-9]+ +[a-z0-9]+ +[a-z]+ +-?[0-9]+ / { to[fi] = $6; wo[fi] = $9; ti[fi] = $10; wi[fi] = $13; }
    END {
      for (i = 1; i <= n; i++) {
        split(S[i], p, ":"); tag[i] = p[1]; K[i] = p[2];
        if (tag[i] ~ /^healthy/) { h++; so += to[i]; si += ti[i]; }
      }
      t0o = so / h; t0i = si / h;
      printf "%-20s %3s  %21s  %15s  %6s  %17s  %s\n", "run", "K", "time oop / ip (us)", "T/T0 oop / ip", " model",
             "vs. model oop/ip", "#wrong";
      for (i = 1; i <= n; i++) {
        if (tag[i] ~ /^healthy/) m = 1;
        else if (tag[i] ~ /^balance/) m = 1 / (1 - X);
        else m = 1 + X / ((1 - X) * 2 * (P - 1) / P) * (1 + 1 / K[i]);
        ro = to[i] / t0o; ri = ti[i] / t0i;
        printf "%-20s %3s  %10.0f / %8.0f  %7.3f / %5.3f  %6.4f  %+7.2f%% / %+5.2f%%  %s/%s\n", tag[i], K[i], to[i], ti[i],
               ro, ri, m, (ro / m - 1) * 100, (ri / m - 1) * 100, wo[i], wi[i];
      }
      print "(T0 = mean of the healthy runs, out-of-place and in-place separately. Model: 1/(1-X) for Balance,";
      print " 1 + X/((1-X)*2(P-1)/P)*(1+1/K) for R2CC-AllReduce. #wrong: the -c 1 check, out-of-place/in-place)";
    }' "${files[@]}"
}
run_logged 07.nccl_tests_r2cc_allreduce_k_sweep main "$@"
