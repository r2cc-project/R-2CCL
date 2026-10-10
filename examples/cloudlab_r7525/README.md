# R2CC experiments on CloudLab r7525

Run the scripts in this directory from `node-1`, one at a time. [r7525_setup.md](r7525_setup.md) describes how to
set up the servers.

| Test | node-1's `mlx5_2` | What it shows | Run time | Log |
|---|---|---|---|---|
| [01](01.hot_repair_to_balance.sh) | cut during the run | real NIC failure during a 4 GiB AllReduce → hot repair → **R2CC-Balance** | ~40 s | [log](logs/01.hot_repair_to_balance.log) |
| [02](02.hot_repair_to_r2cc_allreduce.sh) | cut during the run | same failure → hot repair → **R2CC-AllReduce** | ~40 s | [log](logs/02.hot_repair_to_r2cc_allreduce.log) |
| [03](03.nccl_tests_compare_all.sh) | healthy, then declared failed | nccl-tests, 256 MiB–4 GiB, healthy vs. Balance vs. R2CC-AllReduce, one table | ~3 min | [log](logs/03.nccl_tests_compare_all.log) |
| [04](04.nccl_tests_baseline_healthy.sh) | healthy | nccl-tests (message sizes 8 B–4 GiB), R2CC switched off | ~1.5 min | [log](logs/04.nccl_tests_baseline_healthy.log) |
| [05](05.nccl_tests_balance_unhealthy.sh) | declared failed | nccl-tests (message sizes 8 B–4 GiB), R2CC-Balance | ~2 min | [log](logs/05.nccl_tests_balance_unhealthy.log) |
| [06](06.nccl_tests_r2cc_allreduce_unhealthy.sh) | declared failed | nccl-tests (message sizes 8 B–4 GiB), R2CC-AllReduce | ~2 min | [log](logs/06.nccl_tests_r2cc_allreduce_unhealthy.log) |
| [07](07.nccl_tests_r2cc_allreduce_k_sweep.sh) | healthy, then declared failed | 4 GiB, R2CC-AllReduce with K = 1–16 pipeline chunks against the paper's formula | ~7 min | [log](logs/07.nccl_tests_r2cc_allreduce_k_sweep.log) |
| [08](08.hot_repair_cuda_graph.sh) | cut during the run | the hot repair with the AllReduce replayed from a CUDA graph: time per iteration across the failure and after capturing it again as Balance or R2CC-AllReduce; every element checked as in 01/02 | ~1.5 min | [log](logs/08.hot_repair_cuda_graph.log) |
| [09](09.training_with_nic_failure.sh) | healthy, or cut at update 400 | GPT-2 (124M) training, 1000 updates: upstream NCCL, R2CC, and R2CC with the failure (then Balance or R2CC-AllReduce), compared bit for bit and by test perplexity | ~45 min | [log](logs/09.training_with_nic_failure.log) |

The run times and the reference logs come from this testbed. The times include start-up and restoring the NIC;
01–08 together take about 17 minutes.

*Cut during the run* means that the BlueField drops all traffic of the port while an AllReduce is running, and R2CC
has to detect and repair the failure. *Declared failed* means that the port is never cut; `R2CC_FAILED_NODE` and
`R2CC_FAILED_HCA` make every rank treat it as failed from the start, which is the state after a repair. *Cut at
update 400* is the same cut, made by the training at update 400 and left in place.

## 1. Testbed

### 1.1 Configuration

```
node-K (K = 1, 2, 3)
  NUMA 0:  GPU0  V100S     mlx5_0  10 Gb/s
  NUMA 1:  GPU1  V100S     mlx5_2  10 Gb/s   <- the NIC that fails
                           mlx5_3  10 Gb/s
```

The three NIC ports of a CloudLab r7525 server have different speeds (25, 100 and 100 Gb/s), so
`nic/shape_nics.sh 10` limits every port to 10 Gb/s and `~/topo.xml` declares the same speed. The three NICs are
then equal (1.18–1.23 GB/s each per direction) and bound every collective. The two GPUs sit on different NUMA nodes
and have no NVLink.

### 1.2 What the model in the paper predicts under this testbed

The paper models R2CC-AllReduce in section "Failure-aware Schedule Optimization" (subsection "R2CC AllReduce").
One node loses a fraction X of its bandwidth B; with per-node data D and n nodes, a global AllReduce on (1−Y)D runs
concurrently with a partial AllReduce of the healthy nodes on YD, and Stage 2 broadcasts YD back, with Y = X.

```
T_1  = 2(n−1)/n   · (1−Y)D / ((1−X)B)        global AllReduce
T_2  = 2(n−2)/(n−1) · YD / (XB)              partial AllReduce
T_3  = YD / ((1−X)B)                         broadcast back
T    = max(T_1, T_2) + T_3                   T_nf = 2(n−1)D / (nB)   (no failure)

T_Balance / T_nf = 1/(1−X)                   T / T_nf = 1 + nX / (2(n−1)(1−X))
```

On this testbed the ring has n = 6 members (two GPUs per server), X = 1/3 (one of node-1's three NICs), and the
partial AllReduce runs on the 4 ranks of the healthy servers (T_2 = 0.9 T_nf, hidden behind T_1). The implementation
runs Stage 2 as K pipelined chunks of a Reduce and a Broadcast (`R2CC_AR_STAGE2_CHUNKS`, default 4), which adds one
chunk to T_3, T_3 = (1 + 1/K) · XD / ((1−X)B). The prediction is therefore

- R2CC-Balance, T/T_nf = 1/(1−X) = **1.5**;
- R2CC-AllReduce, T/T_nf = 1 + 0.3 · (1 + 1/K) = **1.375** for K = 4, which approaches the paper's 1 + nX/(2(n−1)(1−X))
  = 1.30 as K grows. Test 07 measures K = 1 to 16.

| 4 GiB AllReduce | prediction | test 03 (in-place) | tests 01/02 |
|---|---|---|---|
| healthy | 1 | 2.03 s, busbw 3.53 GB/s | 2.04 s (01), 2.03 s (02) |
| R2CC-Balance, `mlx5_2` failed | 1.5 | 1.51 (3.05 s, 2.34 GB/s) | 1.49 (3.04 s) |
| R2CC-AllReduce, `mlx5_2` failed, K = 4 | 1.375 | 1.38 (2.79 s, 2.56 GB/s) | 1.38 (2.80 s) |

R2CC-AllReduce gives 9.1% more bandwidth than Balance by the prediction and 8.9–9.4% in test 03.

## 2. Run the tests

- Run everything from `node-1` in `/mydata/R2CC/examples/cloudlab_r7525`, e.g. `./01.hot_repair_to_balance.sh`.
  All nine in a row take about an hour, `for t in ./0[1-9].*.sh; do SAVE_LOG=1 $t; done`.
- One job at a time. Every script refuses to start while another one runs and first restores `mlx5_2` on the
  SmartNIC. `tools/kill.sh` stops leftover processes on all nodes.
- The scripts print to the terminal and never overwrite the reference logs in `logs/`. `SAVE_LOG=1` also saves the
  output to `logs/local/`; 09 always writes its run files to `OUT` (section 5).
- A failed run returns a non-zero exit code. 01, 02 and 08 end with a `[result]` line (0 pass, 2 wrong results,
  3 the cut did not hit a running AllReduce, run again).
- On a new testbed, follow [r7525_setup.md](r7525_setup.md), build nccl-tests with `tools/build_nccl_tests.sh`,
  limit the NICs with `nic/shape_nics.sh 10`, and run `tools/sync.sh` after every rebuild. `REMOTE_HOSTS=node-2`
  restricts a run to two servers.

The failed NIC is always node-1's `mlx5_2`. Tests 01, 02, 08 and 09 cut it on the BlueField with an OVS drop rule
(`nic/disconnect_nic1.sh`, removed by `nic/connect_nic1.sh`); tests 03 and 05–07 only declare it failed.

## 3. Results and analysis

The `mlx5_*_RX` columns are the MB that node-1's ports received in each iteration (`port_rcv_data`).

### 3.1 Test 01 — real NIC failure, hot repair, then R2CC-Balance

`hot_repair/test_hot_repair` runs ten 4 GiB AllReduces on the six GPUs. Four seconds after the start, during
iteration 3, the SmartNIC drops all traffic of node-1's `mlx5_2`. R2CC moves the in-flight transfers to the backup
connection, finishes the AllReduce and then runs R2CC-Balance.

- **Every element of every iteration is checked on every rank.** The inputs are integers small enough that every
  sum is exact, so a GPU kernel can compare all 2^30 output elements with the exact result.
- **`Failure evidence`** shows that the cut hit a running AllReduce and that `mlx5_2` carried nothing afterwards.
- `TEST PASS` needs both. Exit code 0 is a pass, 2 wrong results, 3 a cut that missed a running AllReduce (run
  again).
- `R2CC_TEST_CORRUPT=3` (or `3,nan`) corrupts one output element to test the check, and the run fails with exit
  code 2. This works for 02 and 08 too (`hot_repair/test_common.h`).

From [logs/01.hot_repair_to_balance.log](logs/01.hot_repair_to_balance.log):

```
[testbed] node-1 mlx5_0 egress ratelimit: 10.0 Gbps (nic/shape_nics.sh status shows all ports)
[Rank 0] Iter 2/10 END: OK (elapsed 2040 ms)
[Rank 0] Iter 3/10 START: allreduce 4.00 GiB
[Rank 0] NIC disconnect command completed.        <- mlx5_2 is now black-holed, iteration 3 is in flight
[Rank 0] Iter 3/10 END: OK (elapsed 3539 ms)      <- repaired mid-collective, every element correct
...
[Rank 0] IB RX per-iteration (MB, port_rcv_data *4B):
Iter   Time(ms)   mlx5_0_RX    mlx5_2_RX    mlx5_3_RX
1      2162       2406         2312         2312       <- iteration 1 includes connection setup
2      2040       2406         2312         2312       <- healthy: ~7.0 GB received, 2(P-1)/P x 4 GiB for P = 6
                                                          ranks plus headers, one third per NIC
3      3539       4188         601          2312       <- failure: mlx5_2 stops after 601 MB; the rest of its
                                                          share is migrated to the backup connection on mlx5_0
4      3040       3609         0            3468       <- R2CC-Balance: mlx5_2 unused, the same ~7 GB split
5      3043       3609         0            3468          over the two healthy NICs
...
10     3043       3609         0            3468
[Rank 0] Verification: all 1073741824 elements of each of the 10 iterations checked on all 6 ranks: 0 wrong
[Rank 0] Failure evidence: mlx5_2 cut during iteration 3 (601 of 2312 MB received before the cut; command ran
         4.00-4.75 s, iteration 4.24-7.77 s), no traffic on it in iterations 4-10, all of them completed
[Rank 0] TEST PASS: all AllReduces completed and every element of every iteration is correct, including the one
         the NIC failure hit.
[result] exit=0 PASS
```

An iteration takes 2.04 s with three NICs and 3.04 s with two, 1.49 times as long (formula 1.5). The failover
iteration costs about 1.5 s more than a healthy one.

### 3.2 Test 02 — real NIC failure, hot repair, then R2CC-AllReduce

The same run, but after the repair R2CC-AllReduce runs (`R2CC_AR_AFTER_REPAIR=3`). Its stages are NCCL collectives
on two sub-communicators, created with `ncclCommSplit` in the first AllReduce after the repair.
From [logs/02.hot_repair_to_r2cc_allreduce.log](logs/02.hot_repair_to_r2cc_allreduce.log):

```
Iter   Time(ms)   mlx5_0_RX    mlx5_2_RX    mlx5_3_RX
2      2033       2406         2312         2312       <- healthy, as in test 01
3      2974       3184         1566         2312       <- failure hits during iteration 3, the remainder of
                                                          mlx5_2's share is migrated to mlx5_0
4      3162       3128         0            3005       <- first R2CC-AllReduce: includes ncclCommSplit of the
                                                          two sub-communicators (one-off, ~0.36 s)
5      2824       3129         0            3005       <- steady state: node-1, the degraded server, receives
6      2796       3128         0            3005          ~6.1 GB per iteration instead of the ~7.1 GB of
...                                                       Balance: it takes part in the (1-X) AllReduce and
10     2869       3128         0            3005          receives the broadcast tail, but not the tail's AllReduce
[Rank 0] Verification: all 1073741824 elements of each of the 10 iterations checked on all 6 ranks: 0 wrong
[Rank 0] Failure evidence: mlx5_2 cut during iteration 3 (1566 of 2312 MB received before the cut; command ran
         4.00-5.65 s, iteration 4.32-7.30 s), no traffic on it in iterations 4-10, all of them completed
[Rank 0] TEST PASS: all AllReduces completed and every element of every iteration is correct, including the one
         the NIC failure hit.
```

An iteration takes 2.80 s, 1.38 times the healthy 2.03 s (formula 1.375 for K = 4) and 8% less than with Balance.
node-1 receives 13% less than with Balance, exactly (2(P−1)/P·(1−X) + X) / (2(P−1)/P) = 0.867; node-2 and node-3
carry the tail AllReduce on their otherwise idle `mlx5_2`.

### 3.3 Test 03 — nccl-tests: correctness and a side-by-side table

`all_reduce_perf` runs three times with the same arguments, healthy with R2CC switched off (`R2CC_MODE=0`), then
R2CC-Balance (`R2CC_MODE=2`) and R2CC-AllReduce (`R2CC_MODE=3`) with `mlx5_2` declared failed. With `-c 1`,
nccl-tests checks one more AllReduce of every size element by element (`#wrong`); the timed iterations are not
checked. From [logs/03.nccl_tests_compare_all.log](logs/03.nccl_tests_compare_all.log):

```
===== comparison (-b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum) =====
bytes        | baseline_healthy                 | balance_unhealthy                | r2cc_allreduce_unhealthy
268435456    | 3.36/2.76 (wrong 0/0)            | 2.24/2.31 (wrong 0/0)            | 2.37/2.50 (wrong 0/0)
1073741824   | 3.52/3.30 (wrong 0/0)            | 2.35/2.36 (wrong 0/0)            | 2.56/2.56 (wrong 0/0)
4294967296   | 3.53/3.53 (wrong 0/0)            | 2.35/2.34 (wrong 0/0)            | 2.56/2.56 (wrong 0/0)
(cells: busbw out-of-place/in-place GB/s, then #wrong out-of-place/in-place)
```

All `#wrong` are 0. At 4 GiB, healthy runs at 3.53 GB/s (three NICs), Balance at 2.34–2.35 GB/s (two NICs) and
R2CC-AllReduce at 2.56 GB/s, 1.51 and 1.38 times the healthy time (formula 1.5 and 1.375). Smaller sizes vary more
between runs.

### 3.4 Tests 04–06 — full nccl-tests sweeps

Each runs `all_reduce_perf` over every message size from 8 B to 4 GiB (`-b 8 -e 4G -f 2 -g 1 -c 1 -n 5 -w 2
-d float -o sum`), checked with `-c 1` as in 03:

| Test | Scenario | busbw from 512 MiB to 4 GiB (out-of-place / in-place) | at 4 GiB |
|---|---|---|---|
| 04 | healthy, R2CC switched off (`R2CC_MODE=0`) | 3.49–3.53 / 3.44–3.53 GB/s | 3.53 / 3.53 GB/s |
| 05 | R2CC-Balance, `mlx5_2` failed (`R2CC_MODE=2`) | 2.35–2.36 / 2.35–2.36 GB/s | 2.35 / 2.35 GB/s |
| 06 | R2CC-AllReduce, `mlx5_2` failed (`R2CC_MODE=3`) | 2.50–2.56 / 2.50–2.56 GB/s | 2.56 / 2.56 GB/s |

Below 16 MiB (`R2CC_AR_MIN_BYTES`) R2CC-AllReduce falls back to Balance, with one warning per rank. Single sizes up
to 256 MiB are sometimes slower in one pass; from 512 MiB on, runs differ by at most about 0.1 GB/s. The scripts
take nccl-tests arguments and `NCCL_TEST_BIN` for other collectives:

```bash
./06.nccl_tests_r2cc_allreduce_unhealthy.sh -b 1G -e 4G -f 4 -d half -o prod
NCCL_TEST_BIN=all_gather_perf ./05.nccl_tests_balance_unhealthy.sh -b 64M -e 1G -f 2 -d float
R2CC_AR_STAGE2_CHUNKS=8 ./06.nccl_tests_r2cc_allreduce_unhealthy.sh -b 4G -e 4G     # K = 8 instead of 4
```

### 3.5 Test 07 — the pipeline depth K of R2CC-AllReduce against the paper's formula

`all_reduce_perf` at 4 GiB with `mlx5_2` declared failed, in nine configurations, healthy, Balance, R2CC-AllReduce
with K = 1, 2, 4, 8 and 16, then healthy and Balance again to show drift. Every time is divided by the mean of the
two healthy runs and printed next to the formula of section 1.2.
From [logs/07.nccl_tests_r2cc_allreduce_k_sweep.log](logs/07.nccl_tests_r2cc_allreduce_k_sweep.log):

```
run                    K     time oop / ip (us)    T/T0 oop / ip   model   vs. model oop/ip  #wrong
healthy_1              -     2028245 /  2029138    1.000 / 1.000  1.0000    -0.03% / -0.00%  0/0
balance_1              -     3039966 /  3039571    1.498 / 1.498  1.5000    -0.11% / -0.14%  0/0
r2cc_allreduce_K1      1     3243624 /  3244620    1.599 / 1.599  1.6000    -0.08% / -0.06%  0/0
r2cc_allreduce_K2      2     2958525 /  2947277    1.458 / 1.452  1.4500    +0.57% / +0.17%  0/0
r2cc_allreduce_K4      4     2791287 /  2793662    1.376 / 1.377  1.3750    +0.06% / +0.13%  0/0
r2cc_allreduce_K8      8     2720541 /  2716429    1.341 / 1.339  1.3375    +0.26% / +0.09%  0/0
r2cc_allreduce_K16    16     2695495 /  2692850    1.329 / 1.327  1.3187    +0.74% / +0.63%  0/0
healthy_2              -     2029505 /  2029159    1.000 / 1.000  1.0000    +0.03% / +0.00%  0/0
balance_2              -     3040734 /  3041923    1.499 / 1.499  1.5000    -0.08% / -0.06%  0/0
```

Every configuration is within 0.74% of the formula and every check is 0; the runs before and after the sweep
differ by at most 0.1%. With K = 1 R2CC-AllReduce is slower than Balance (1.60); from K = 2 on it is faster and
approaches the K → ∞ value 1.30.

## 4. CUDA graphs

This is the CUDA Graphs compatibility experiment added to the appendix during shepherding. `08.hot_repair_cuda_graph.sh`
runs `hot_repair/test_hot_repair_graph` twice with the failure of 01/02 in iteration 3. The graph captured at the
start is replayed in every iteration; after iteration 6 the AllReduce is captured again, as R2CC-Balance in run 1 and
as R2CC-AllReduce in run 2, which first needs one AllReduce outside the capture (`--eager`). Every element of every
AllReduce is checked as in 01, outside the timed part.
From [logs/08.hot_repair_cuda_graph.log](logs/08.hot_repair_cuda_graph.log):

```
===== CUDA Graphs: 4 GiB AllReduce on all GPUs, mlx5_2 of node-1 cut during the run =====
Replayed graph (4 GiB AllReduce)    Time / iter   Re-capture overhead*
Pre-failure graph, healthy               2.03 s   --
Pre-failure graph, HotRepair             4.07 s   --
Re-captured as R2CC-Balance              3.04 s   1.7 ms
Re-captured as R2CC-AllReduce            2.79 s   0.33 s
* the time of the iteration that performs the re-capture, including the eager collective that R2CC-AllReduce
  needs to create its sub-communicators, minus the time of an iteration replayed from the new graph

run 1/2, re-captured as R2CC-Balance:
  Verification: all 1073741824 elements of each of the 10 iterations checked on all 6 ranks: 0 wrong
  Failure evidence: mlx5_2 cut during iteration 3 (847 of 2312 MB received before the cut; command ran 4.00-4.83 s, iteration 4.11-7.43 s), no traffic on it in iterations 4-10, all of them completed
  TEST PASS: all AllReduces completed and every element of every iteration is correct, including the one the NIC failure hit.
run 2/2, re-captured as R2CC-AllReduce:
  Verification: all 1073741824 elements of each of the 10 iterations and of the AllReduce before re-capture checked on all 6 ranks: 0 wrong
  Failure evidence: mlx5_2 cut during iteration 3 (674 of 2312 MB received before the cut; command ran 4.00-4.68 s, iteration 4.11-7.58 s), no traffic on it in iterations 4-10, all of them completed
  TEST PASS: all AllReduces completed and every element of every iteration is correct, including the one the NIC failure hit.
[result] exit=0 PASS
```

- **Every element of every AllReduce is correct in both runs**, including the replays across the failure.
- **The pre-failure graph keeps replaying after the repair.** It still has the healthy channel layout, so the backup
  connection on `mlx5_0` carries the whole share of `mlx5_2` (4.07 s).
- **Balance and R2CC-AllReduce are chosen when a collective is enqueued**, so they need a new capture, 1.7 ms for
  Balance. R2CC-AllReduce adds 0.33 s for its AllReduce outside the capture.

## 5. Training through a NIC failure

This is the training-quality experiment added during shepherding (summarized in the evaluation section, details in
the appendix). `09.training_with_nic_failure.sh` trains GPT-2 (124M) on WikiText-103 with PyTorch DDP on the six
GPUs for 1000 updates (`training/train.py`), with a fixed seed and deterministic kernels. Every update runs one
475 MiB gradient AllReduce. Four runs use the same seed (default 42):

| Run | Library | node-1's `mlx5_2` | Schedule after the hot repair |
|---|---|---|---|
| VNF | upstream NCCL 2.23.4 | healthy | – |
| NF | R2CC | healthy | – |
| BALF | R2CC | cut at update 400 | R2CC-Balance |
| ARF | R2CC | cut at update 400 | R2CC-AllReduce |

The library is not told which NIC fails. `training/compare.py` compares every run with VNF by the training loss of
every update, the SHA-256 of the AllReduce outputs of updates 400–408 and of the parameters, the error of those
AllReduces against an FP64 sum of the inputs (E_rel, Euclidean norm over all elements), and the test perplexity.
From [logs/09.training_with_nic_failure.log](logs/09.training_with_nic_failure.log):

```
[09] VNF_42: done in 10 min, test PPL 75.56742342917366
[09] NF_42: done in 10 min, test PPL 75.56742342917366
[09] BALF_42: done in 11 min, test PPL 75.57225077094003
[09] ARF_42: done in 11 min, test PPL 75.58452098091739

===== seed 42: every run against VNF (NCCL 2.23.4, no failure) =====
run   NIC down    identical to VNF:                         max      max  test PPL   vs VNF
      from        reduced gradient  training loss    |loss-VNF|    E_rel
VNF   -           -                 -                         -  5.8e-08   75.5674        -
NF    -           updates 400-408   updates 1-1000            0  5.8e-08   75.5674  +0.000%
BALF  update 401  updates 400-401   updates 1-402      1.31e-03  5.8e-08   75.5723  +0.006%
ARF   update 401  updates 400-401   updates 1-402      1.76e-03  5.8e-08   75.5845  +0.023%
...
===== Training quality: GPT-2 (124M), WikiText-103, mlx5_2 of node-1 cut at update 400 (seed 42) =====
Condition                  Test PPL   Max. paired Δ*
NCCL 2.23.4, no failure      75.567   --
R2CC, no failure             75.567   0.000%
R2CC-Balance, failure        75.572   +0.006%
R2CC-AllReduce, failure      75.585   +0.023%
...
===== checks =====
yes  seed 42: NF is bit-identical to VNF: training loss of every update, test loss, reduced gradients, parameters
yes  seed 42: BALF and ARF lost mlx5_2 during the run
yes  seed 42: BALF and ARF: the AllReduce that the failure hit gives the same reduced gradient as VNF
yes  seed 42: BALF and ARF: the training loss is identical to VNF up to the failure
yes  seed 42: all runs: every AllReduce of updates 400-408 is identical on all ranks and has E_rel < 1e-06
yes  seed 42: all runs: the parameters are identical on all ranks at every check point
RESULT: all checks passed
```

- **Without a failure, R2CC is identical to upstream NCCL bit for bit.**
- **The hot repair returns the same bits.** The cut takes effect before the AllReduce of update 401, which R2CC
  repairs; its output equals VNF's, so the loss is identical up to update 402. Update 401 takes about 1.4 s instead
  of 0.5 s, and training continues without a restart.
- **From update 402 on** the schedules sum the gradients in a different order, so the results differ only by FP32
  rounding (E_rel 5.8e-8, as for NCCL). The loss differs by at most 1.76e-3 and the test perplexity by at most
  +0.023%.

These runs are bit-identical to the seed-42 runs in the paper. The paper reports seeds 42, 43 and 44, for which
`training/compare.py` prints

```
Condition                  Test PPL   Max. paired Δ*
NCCL 2.23.4, no failure      76.085   --
R2CC, no failure             76.085   0.000%
R2CC-Balance, failure        76.085   +0.008%
R2CC-AllReduce, failure      76.086   +0.023%
```

The runs use PyTorch 2.4.1 built against the system NCCL, so each run loads R2CC or upstream NCCL 2.23.4 from
`LD_LIBRARY_PATH`, and WikiText-103 tokenized with the GPT-2 BPE (SHA-256 in `manifest.json`), all under
`/proj/softmeasure-PG0/r2cc_ae`. The files of every run go to `OUT`, printed at the start.

## 6. Directory layout

```
0[1-9].*.sh     the nine tests
common.sh       shared settings: mpirun line, failure model, check_idle, NIC restore, nccl-tests runner, table
nodes.sh        the servers of the experiment (node-1 + REMOTE_HOSTS) and their addresses
hot_repair/     test_hot_repair.cc + run_hot_repair.sh (01/02), test_hot_repair_graph.cc (08), test_common.h, binaries
training/       train.py (one training run of 09), compare.py (its table and checks)
nic/            disconnect_nic1.sh / connect_nic1.sh (OVS drop rule), shape_nics.sh (NIC rate limits), setup checks
xml/            NCCL topology dumper and the equal-speed topo.xml
setup/          02.setup_network_and_nic.sh, the Phase 2 script of r7525_setup.md
tools/          kill.sh, sync.sh, stress_test.sh, build_nccl_tests.sh
logs/           terminal output of one run of each test
r7525_setup.md  how to set up the r7525 servers
```
