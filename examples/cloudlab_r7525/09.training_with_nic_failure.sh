#!/usr/bin/env bash
# GPT-2 (124M) training through a real NIC failure, against failure-free training (README.md, section 3.7). Four
# runs of training/train.py with the same seed, each 1000 optimizer updates of data-parallel training on the six
# GPUs (WikiText-103, global batch 48 x 1024 tokens, deterministic kernels):
#   VNF    upstream NCCL 2.23.4, no failure
#   NF     R2CC, no failure
#   BALF   R2CC; node-1's mlx5_2 is cut at update 400 and stays down until the run ends; R2CC-Balance after the
#          hot repair (R2CC_AR_AFTER_REPAIR=2)
#   ARF    the same with R2CC-AllReduce after the hot repair (R2CC_AR_AFTER_REPAIR=3)
#   The library is not told which NIC fails (no R2CC_FAILED_*). training/compare.py then compares every run with
#   VNF: the training loss of every update, the output of the gradient AllReduce of updates 400-408 (bit for bit,
#   and against an FP64 sum of its inputs), the parameters and the test perplexity, and prints the test perplexity
#   table of the paper. A run that fails, or whose train_log.csv does not grow for STALL_S seconds (default 300),
#   is stopped and mlx5_2 is restored. About 45 minutes per seed.
#
# Usage: ./09.training_with_nic_failure.sh
#   SEEDS="42 43 44" runs the three seeds of the paper (default 42; about 2.3 hours).
#   Uses the Python environment, the tokenized WikiText-103 and the upstream NCCL build under AE_ROOT (default
#   /proj/softmeasure-PG0/r2cc_ae, see README.md section 3.7). The files of every run (train_log.csv, summary.json,
#   stdout.log) go to OUT (default /mydata/r2cc_training/<date and time>).
#   Nothing is written to logs/ by default; SAVE_LOG=1 also saves the terminal output to logs/local/.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "${EXAMPLE_DIR}"
check_idle
ensure_nic_connected
unset R2CC_FAILED_NODE R2CC_FAILED_HCA R2CC_FAILED_NIC_COUNT   # set by common.sh; the library has to find the failure

AE_ROOT="${AE_ROOT:-/proj/softmeasure-PG0/r2cc_ae}"
PY="${AE_ROOT}/venv/bin/python"
DATA="${AE_ROOT}/data"
UPSTREAM_LIB="${AE_ROOT}/nccl_vanilla/lib"
R2CC_LIB="${REPO_ROOT}/build/lib"
SEEDS="${SEEDS:-42}"
OUT="${OUT:-/mydata/r2cc_training/$(date +%Y%m%d_%H%M%S)}"
STALL_S="${STALL_S:-300}"
for f in "${PY}" "${DATA}/manifest.json" "${UPSTREAM_LIB}/libnccl.so.2" "${R2CC_LIB}/libnccl.so.2"; do
  [[ -e "${f}" ]] || { echo "missing: ${f}" >&2; exit 1; }
done
# Every rank runs training/train.py from its own copy of the repository (README.md section 2: tools/sync.sh).
want="$(sha256sum training/train.py | cut -d' ' -f1)"
for h in "${REMOTE_HOST_LIST[@]}"; do
  if [[ "$(ssh -o BatchMode=yes "${h}" "sha256sum ${EXAMPLE_DIR}/training/train.py" 2>/dev/null | cut -d' ' -f1)" != "${want}" ]]; then
    echo "${h}:${EXAMPLE_DIR}/training/train.py differs from node-1's; run tools/sync.sh" >&2
    exit 1
  fi
done
export PATH="${AE_ROOT}/venv/bin:${PATH}"

# Stops a stalled run on every node and restores mlx5_2 (not tools/kill.sh, which would also stop this script).
stop_run() {
  local pat='[t]raining/train\.py|[m]pirun -np|[o]rted -mca' h
  pkill -9 -f "${pat}" || true
  for h in "${REMOTE_HOST_LIST[@]}"; do ssh -o BatchMode=yes "${h}" "pkill -9 -f '${pat}' || true" || true; done
  ./nic/connect_nic1.sh > /dev/null 2>&1 || true
}

run_one() {   # run_one <condition> <seed>
  local cond="$1" seed="$2" name="$1_$2" lib="${R2CC_LIB}" fail=400 after=2 what
  case "${cond}" in
    VNF)  lib="${UPSTREAM_LIB}"; fail=0; what="upstream NCCL 2.23.4, no failure" ;;
    NF)   fail=0; what="R2CC, no failure" ;;
    BALF) what="R2CC, mlx5_2 cut at update 400, then R2CC-Balance" ;;
    ARF)  after=3; what="R2CC, mlx5_2 cut at update 400, then R2CC-AllReduce" ;;
  esac
  mkdir -p "${OUT}/${name}"
  echo "[09] ${name}: ${what}"
  ./nic/connect_nic1.sh > /dev/null 2>&1 || true
  local t0=${SECONDS}
  "${MPIRUN_BASE[@]}" --bind-to none \
    -x PATH -x "LD_LIBRARY_PATH=${lib}:/usr/local/cuda/lib64:/mydata/openMpi/lib" -x "EXPECT_LIBNCCL=${lib}/libnccl.so.2" \
    -x MASTER_ADDR=10.10.1.1 -x MASTER_PORT=29561 -x GLOO_SOCKET_IFNAME=eno33np0 \
    -x OMP_NUM_THREADS=8 -x MKL_NUM_THREADS=8 -x OPENBLAS_NUM_THREADS=8 \
    -x CUBLAS_WORKSPACE_CONFIG=:4096:8 -x PYTHONHASHSEED=1337 -x PYTHONUNBUFFERED=1 \
    -x R2CC_MODE=0 -x "R2CC_AR_AFTER_REPAIR=${after}" -x NCCL_IB_TIMEOUT=16 -x NCCL_IB_RETRY_CNT=1 \
    -x "NCCL_DEBUG=${NCCL_DEBUG:-WARN}" \
    "${PY}" training/train.py --name "${name}" --seed "${seed}" --fail-at-step "${fail}" --data "${DATA}" --out "${OUT}" \
    > "${OUT}/${name}/stdout.log" 2>&1 &
  local pid=$! csv="${OUT}/${name}/train_log.csv" last=0 quiet=0 now rc
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 15
    now=$(stat -c %s "${csv}" 2>/dev/null || echo 0)
    if (( now > last )); then last=${now}; quiet=0; else quiet=$((quiet + 15)); fi
    if (( quiet >= STALL_S )); then
      echo "[09] ${name}: no progress for ${quiet} s; stopping it (output: ${OUT}/${name}/stdout.log)"
      kill -9 "${pid}" 2>/dev/null || true
      stop_run
      return 1
    fi
  done
  set +e; wait "${pid}"; rc=$?; set -e
  ./nic/connect_nic1.sh > /dev/null 2>&1 || true
  if [[ ${rc} -ne 0 || ! -f "${OUT}/${name}/summary.json" ]]; then
    echo "[09] ${name}: failed (exit ${rc}; output: ${OUT}/${name}/stdout.log)"
    return 1
  fi
  echo "[09] ${name}: done in $(( (SECONDS - t0) / 60 )) min, test PPL $(grep -o '"test_ppl": [0-9.]*' "${OUT}/${name}/summary.json" | cut -d' ' -f2)"
}

main() {
  local seed cond rc=0
  mkdir -p "${OUT}"
  print_nic_limit
  echo "[09] seeds ${SEEDS}; the files of every run go to ${OUT}"
  for seed in ${SEEDS}; do
    for cond in VNF NF BALF ARF; do
      run_one "${cond}" "${seed}" || rc=1
    done
  done
  echo
  "${PY}" training/compare.py "${OUT}" ${SEEDS} || rc=1
  return "${rc}"
}
run_logged 09.training_with_nic_failure main
