# R2CC experiments on CloudLab r7525

This document describes the nine ready-to-run R2CC tests in this directory and the terminal output of one run of
each on three CloudLab `r7525` servers (`node-1`, `node-2`, `node-3`), kept in [`logs/`](logs/) for readers who
cannot run them. With every NIC rate-limited to the same speed, the tests check both sides of the paper: that
R2CC is correct (a real NIC failure is repaired in the middle of a collective, the results are checked element by
element, each NIC carries the traffic the schedule assigns to it) and that R2CC-Balance and R2CC-AllReduce reach
the bandwidth the model predicts. Test 09 trains GPT-2 through a real NIC failure and compares the run with
upstream NCCL. The document contains:

1. [Testbed](#1-testbed) — the servers as CloudLab provides them, the rate-limited configuration used for the
   results, and what the model predicts.
2. [Run the tests](#2-run-the-tests) — prerequisites, how the scripts behave, where the output goes.
3. [Results and analysis](#3-results-and-analysis) — for every test, the key lines of the saved log, what
   happened, how the NIC traffic changed and how the result compares with the model.
4. [CUDA graphs](#4-cuda-graphs) — the hot-repair test with its AllReduce captured in a CUDA graph (test 08).
5. [Training through a NIC failure](#5-training-through-a-nic-failure) — GPT-2 trained with and without the
   failure, against upstream NCCL (test 09).
6. [Directory layout](#6-directory-layout).

Bringing the machines up is described separately in [r7525_setup.md](r7525_setup.md).

| Test | node-1's `mlx5_2` | What it shows | Run time | Reference log (three servers, NICs at 10 Gb/s) |
|---|---|---|---|---|
| `01.hot_repair_to_balance.sh` | cut during the run | real NIC failure during a 4 GiB AllReduce → hot repair → **R2CC-Balance** | ~40 s | [logs/01.hot_repair_to_balance.log](logs/01.hot_repair_to_balance.log) |
| `02.hot_repair_to_r2cc_allreduce.sh` | cut during the run | same failure → hot repair → **R2CC-AllReduce** | ~40 s | [logs/02.hot_repair_to_r2cc_allreduce.log](logs/02.hot_repair_to_r2cc_allreduce.log) |
| `03.nccl_tests_compare_all.sh` | healthy, then declared failed | nccl-tests, 256 MiB–4 GiB, healthy vs. Balance vs. R2CC-AllReduce, one table | ~3 min | [logs/03.nccl_tests_compare_all.log](logs/03.nccl_tests_compare_all.log) |
| `04.nccl_tests_baseline_healthy.sh` | healthy | full nccl-tests sweep (8 B–4 GiB), R2CC switched off | ~1.5 min | [logs/04.nccl_tests_baseline_healthy.log](logs/04.nccl_tests_baseline_healthy.log) |
| `05.nccl_tests_balance_unhealthy.sh` | declared failed | full sweep, R2CC-Balance | ~2 min | [logs/05.nccl_tests_balance_unhealthy.log](logs/05.nccl_tests_balance_unhealthy.log) |
| `06.nccl_tests_r2cc_allreduce_unhealthy.sh` | declared failed | full sweep, R2CC-AllReduce | ~2 min | [logs/06.nccl_tests_r2cc_allreduce_unhealthy.log](logs/06.nccl_tests_r2cc_allreduce_unhealthy.log) |
| `07.nccl_tests_r2cc_allreduce_k_sweep.sh` | healthy, then declared failed | 4 GiB, R2CC-AllReduce with K = 1–16 pipeline chunks against the model | ~7 min | [logs/07.nccl_tests_r2cc_allreduce_k_sweep.log](logs/07.nccl_tests_r2cc_allreduce_k_sweep.log) |
| `08.hot_repair_cuda_graph.sh` | cut during the run | the hot repair with the AllReduce replayed from a CUDA graph: time per iteration across the failure and after capturing it again as Balance or R2CC-AllReduce; every element checked as in 01/02 | ~1.5 min | [logs/08.hot_repair_cuda_graph.log](logs/08.hot_repair_cuda_graph.log) |
| `09.training_with_nic_failure.sh` | healthy, or cut at update 400 | GPT-2 (124M) training, 1000 updates: upstream NCCL, R2CC, and R2CC with the failure (then Balance or R2CC-AllReduce), compared bit for bit and by test perplexity | ~45 min | [logs/09.training_with_nic_failure.log](logs/09.training_with_nic_failure.log) |

Run times were measured on this testbed and include start-up and restoring the NIC; 01–08 together take about
17 minutes.

*Cut during the run*: the BlueField starts dropping all traffic of the port while an AllReduce is running, and
R2CC has to detect and repair the failure. *Declared failed*: the port is never cut; `R2CC_FAILED_NODE` and
`R2CC_FAILED_HCA` make every rank treat it as failed from the start, which is the steady state after a repair.
*Cut at update 400*: the same cut, made by the training at update 400 and left in place until the run ends.

## 1. Testbed

### 1.1 The servers as CloudLab provides them

```
node-K (K = 1, 2, 3; identical servers)
  NUMA 0:  GPU0  V100S, PCIe 3.0 x16     mlx5_0   25 Gb/s  ConnectX-5 port, eno33np0 (also bootstrap / MPI)
  NUMA 1:  GPU1  V100S, PCIe 3.0 x16     mlx5_2  100 Gb/s  BlueField-2 port 0, ens5f0np0   <- the NIC that "fails"
                                         mlx5_3  100 Gb/s  BlueField-2 port 1, ens5f1np1
  GPU0 <-> GPU1: no NVLink, peer-to-peer traffic crosses PCIe and the link between the two sockets
  all nine ports are in one CloudLab LAN; node-K uses 10.10.1.(3K-2), 10.10.2.(3K-1) and 10.10.3.(3K)
```

- The two GPUs of a server sit on **different NUMA nodes and have no NVLink**, so every intra-node hop of a ring
  goes through PCIe and the link between the sockets. Each server has three usable ports: **one 25G and two
  100G**.
- A single 100G port is more than the GPUs of these servers can drive; **plain NCCL would simply pick one 100G
  NIC** and there would be nothing to fail over between. `xml/cloudlab_dump_topo.sh` therefore writes a topology
  file (`~/topo.xml`, passed with `NCCL_TOPO_FILE`) in which **every NIC is declared at 10 Gb/s**. NCCL then
  builds one ring per NIC and splits the data equally over the three. This is what makes the "one of three NICs
  fails" scenario of the paper possible on these servers.

### 1.2 Rate-limited configuration (used for all results below)

`nic/shape_nics.sh 10` limits the transmit rate of every port of every server to 10 Gb/s (ETS rate limit: on the
host ConnectX-5 for all eight traffic classes, on the BlueField ports for traffic class 0, set on the SmartNIC):

```
node-K
  NUMA 0:  GPU0     mlx5_0  10 Gb/s
  NUMA 1:  GPU1     mlx5_2  10 Gb/s   <- the NIC that "fails"
                    mlx5_3  10 Gb/s
  ~/topo.xml declares 10 Gb/s for every NIC: the declared and the real speeds now agree
```

- The three NICs are now really equal. Measured with NCCL over GPUDirect RDMA through a single NIC: 1.18 GB/s
  (`mlx5_0`) and 1.23 GB/s (`mlx5_2`, `mlx5_3`) per direction.
- A healthy ring needs about 3.5 GB/s per GPU and direction, a third of what the GPU side sustains, so the NICs
  bound every collective, which is the setting of the paper's model.
- Only transmitted traffic is limited. Every port is limited, so every link is bounded by its sender, and in the
  collectives of these tests each ring hop has one sender per receiver. The configuration is not meant for
  many-to-one traffic.

### 1.3 What the model predicts

The quantities of the model on this testbed:

- N = 3 servers and P = 6 ranks, one per GPU, all in one ring.
- node-1 is the degraded server: it lost one of its three equal NICs, a fraction **X = 1/3** of its network
  bandwidth. The H = 4 ranks of node-2 and node-3 are on healthy servers.
- K is the number of pipelined chunks of R2CC-AllReduce's Stage 2 (`R2CC_AR_STAGE2_CHUNKS`, default 4).

Let D be the message size and B the network bandwidth of a healthy server. A ring AllReduce moves 2(P−1)/P · D
into and out of every server, so the healthy time is T0 = 2(P−1)/P · D/B = 5/3 · D/B. After the failure node-1
has (1−X)B left and bounds every schedule:

- **R2CC-Balance** moves the same data over (1−X)B: T/T0 = 1/(1−X) = **1.5**.
- **R2CC-AllReduce** has the stage structure of the paper, T ≈ max(T_global, T_partial) + T_tail (section 3.2
  describes the stages):
  - Stage 1: the AllReduce of the first (1−X)D on all P ranks runs over node-1's (1−X)B and takes
    T_global = 2(P−1)/P · (1−X)D / ((1−X)B) = T0. At the same time the partial AllReduce of the last XD among
    the H healthy ranks uses the NIC slot node-1 lost, XB on each healthy server:
    T_partial = 2(H−1)/H · XD / (XB) = 0.9 T0, hidden behind T_global.
  - Stage 2: node-1 sends its part of the tail (Reduce onto a helper rank) and receives the reduced tail
    (Broadcast), XD in each direction over (1−X)B. Cut into K pipelined chunks, the two directions overlap
    except for one chunk: T_tail ≈ (1 + 1/K) · XD / ((1−X)B).

  Together: **T/T0 ≈ 1 + X / ((1−X) · 2(P−1)/P) · (1 + 1/K) = 1 + 0.3 · (1 + 1/K)**, 1.375 for K = 4.

This is the paper's model written for this testbed, with two differences. The ring has P = 6 members, two per
server, so its factor is 2(P−1)/P where the paper's model, with one member per node, has 2(n−1)/n. And the
paper's Stage 2 term XD/((1−X)B) is the limit K → ∞ of the pipelined tail (1.30 here). The paper's formula
1 + nX/(2(n−1)(1−X)) with n = 3 servers also gives 1.375; on this testbed that number is the value for K = 4.
Test 07 (section 3.5) measures K = 1 to 16.

| 4 GiB AllReduce | time relative to healthy, model | measured (nccl-tests, test 03, in-place) | measured (hot repair, tests 01/02) |
|---|---|---|---|
| healthy | 1 | 2.03 s, busbw 3.53 GB/s | 2.04 s (01), 2.03 s (02) |
| R2CC-Balance, `mlx5_2` failed | 1.5 | 1.51 (3.05 s, 2.34 GB/s) | 1.49 (3.04 s) |
| R2CC-AllReduce, `mlx5_2` failed, K = 4 | 1.375 | 1.38 (2.79 s, 2.56 GB/s) | 1.38 (2.80 s) |

With K = 4, R2CC-AllReduce gives 9.1% more bandwidth than R2CC-Balance in the model and 8.9% (out-of-place) and
9.4% (in-place) in test 03. The model and all
results here are for three servers. The scripts also run on two (`REMOTE_HOSTS=node-2`, section 2); there the
healthy side is a single server, the partial AllReduce runs inside it, and the default schedule runs it after the
AllReduce of all ranks (`R2CC_AR_SCHEDULE=2`).

## 2. Run the tests

- Setup: [r7525_setup.md](r7525_setup.md) (CloudLab profile, SmartNIC firmware, network, `~/topo.xml`, NIC rate
  limits). For 03–07 build nccl-tests once with `tools/build_nccl_tests.sh`. 09 uses the Python environment, the
  data and the upstream NCCL build described in section 5.
- Everything is run from `node-1` (`cd /mydata/R2CC/examples/cloudlab_r7525 && ./01.hot_repair_to_balance.sh`).
  The scripts use every server of the experiment: `nodes.sh` takes `node-2`, `node-3`, ... from `/etc/hosts`,
  and `REMOTE_HOSTS=node-2` restricts a run to two servers. `/mydata` is a per-node copy, so after any rebuild
  run `tools/sync.sh`.
- Limit the NICs first: `nic/shape_nics.sh 10` (`status` prints the limits, `off` removes them). Every test prints
  the limit of node-1's `mlx5_0` in a `[testbed]` line. The results in section 3 were measured with these limits.
- One multi-node job at a time. Every script refuses to start while another one is running (`check_idle`
  in `common.sh`) and **every script first restores `mlx5_2` on the SmartNIC** (removes the OVS drop rule),
  so a killed run cannot leave the cluster degraded. `tools/kill.sh` stops leftover processes on all nodes.
- The scripts only print to the terminal; nothing is written to disk by default, so a local run can never
  overwrite the nine reference logs in `logs/`. `SAVE_LOG=1 ./04.nccl_tests_baseline_healthy.sh` additionally
  saves the complete output to `logs/local/<NN>.<name>.log` (git-ignored; `LOG_DIR` changes the directory).
- Every script returns a non-zero exit code when its run fails. For 01, 02 and 08 this includes a wrong element
  and a cut that did not hit a running AllReduce (section 3.1); all three end with a `[result]` line. 09 also fails
  when one of its checks does not hold (section 5).

In every scenario the failed NIC is **node-1's `mlx5_2`**. Tests 01, 02, 08 and 09 cut it during the run on the
BlueField (`nic/disconnect_nic1.sh` installs an OVS drop rule for the port, `nic/connect_nic1.sh` removes it);
tests 03 and 05–07 only declare it failed (`R2CC_FAILED_NODE=0`, `R2CC_FAILED_HCA=mlx5_2`), which gives the
degraded topology after a repair from the start of the run, without a disconnect.

## 3. Results and analysis

Each subsection quotes the key lines of the saved log of one run and explains what happened. All runs: three
servers, six GPUs, every NIC at 10 Gb/s. The `mlx5_*_RX` columns are the MB received by each of node-1's ports
during one iteration (`port_rcv_data`, including packet headers).

### 3.1 Test 01 — real NIC failure, hot repair, then R2CC-Balance

`hot_repair/test_hot_repair` runs 10 AllReduces of 4 GiB (float, sum) on the six GPUs. Four seconds after the
start, during iteration 3, the SmartNIC silently starts dropping all traffic of node-1's `mlx5_2`. The library
detects the stalled connection, live-migrates the in-flight transfers to the backup connection and finishes the
collective; from the next collective on it runs R2CC-Balance (`R2CC_AR_AFTER_REPAIR=2`). The per-iteration table
at the end shows the MB received by each of node-1's ports.

How the run is checked:

- **Every element of every iteration, on every rank.** Each input element is a pseudo-random integer that
  depends on the rank, the element index and the iteration, small enough (below 2^21 with six ranks) that every
  partial sum is exact in float. The correct output is therefore known exactly, and a value that is missing a
  contribution, comes from another position or from another iteration does not match it. After every
  AllReduce, outside the timed part, a GPU kernel compares all 2^30 output elements with the exact sum; NaN and
  Inf never match. A mismatch prints the rank, the iteration, the number of wrong elements and the first wrong
  one with its value and the expected value; the `Verification` line gives the total.
- **The failure hit a running AllReduce.** The `Failure evidence` line names the iteration in which `mlx5_2`
  received only part of its share, the time the disconnect command ran next to the time span of that iteration,
  and confirms that no traffic used `mlx5_2` afterwards.
- `TEST PASS` requires both. The exit code is 0 for PASS, 2 for wrong results, 3 when the cut did not hit a
  running AllReduce (for example it fell between two iterations; run again) and anything else for an abort; the
  script passes it on and prints it in the last line.
- The checker can be tested: `R2CC_TEST_CORRUPT=3 ./01.hot_repair_to_balance.sh` adds one to one output element
  of rank 0 in iteration 3 before the check, `R2CC_TEST_CORRUPT=3,nan` writes NaN there. Both runs end in
  `TEST FAIL: Verification failed ...` with exit code 2. The same works for 02 and 08, which use the same check
  (`hot_repair/test_common.h`).

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

An iteration takes 2.04 s with three NICs and 3.04 s with two, 1.49 times as long, against the model's
1/(1−X) = 1.5: the degraded server bounds the ring, and it now sends and receives the same data over two NICs
instead of three. The failover iteration costs about 1.5 s more than a healthy one. (NCCL without R2CC would hang in iteration 3 until `NCCL_IB_TIMEOUT`/retry expire
and then abort.)

### 3.2 Test 02 — real NIC failure, hot repair, then R2CC-AllReduce

Identical run, but after the repair the library switches to R2CC-AllReduce (`R2CC_AR_AFTER_REPAIR=3`).
With X = failed NICs / NICs per server = 1/3, every AllReduce becomes:

- **Stage 1** — an AllReduce of the first (1−X) of the buffer on all six ranks, over node-1's remaining NICs
  (`mlx5_0`, `mlx5_3`), and at the same time the partial AllReduce of the last X among node-2 and node-3, on a
  sub-communicator that only uses their `mlx5_2`: the NIC slot node-1 lost, which the first AllReduce leaves idle.
- **Stage 2** — the tail X is cut into K = 4 chunks (`R2CC_AR_STAGE2_CHUNKS`); for each chunk a Reduce of
  node-1's data onto a helper rank on node-2 (whose input is the partial result of Stage 1) is pipelined with a
  Broadcast of the finished chunk to all ranks.

The stages are NCCL collectives on two sub-communicators created with `ncclCommSplit`, issued on separate CUDA
streams ordered by events; the sub-communicators are created on the first AllReduce that runs in this mode.
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

An iteration now takes 2.80 s (median of iterations 5–10), 1.38 times the healthy 2.03 s against 1.375 in the
model for K = 4, and 8% less than with Balance (3.04 s). The traffic is the one the paper describes: node-1
receives 13% less than with Balance, which is exactly (2(P−1)/P·(1−X) + X) / (2(P−1)/P) = 0.867 for P = 6 ranks;
node-2 and node-3 absorb the tail AllReduce on their otherwise idle `mlx5_2`.

### 3.3 Test 03 — nccl-tests: correctness and a side-by-side table

`03.nccl_tests_compare_all.sh` runs `all_reduce_perf` three times with identical arguments
(`-b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum`): the healthy cluster with R2CC switched off
(`R2CC_MODE=0`), R2CC-Balance (`R2CC_MODE=2`) and R2CC-AllReduce (`R2CC_MODE=3`) with `mlx5_2` declared failed,
and prints one table. `R2CC_MODE=0` is this repository's library with R2CC switched off; it is the healthy
reference for the times relative to healthy, not a comparison with upstream NCCL. With `-c 1`, after the timed
iterations of every size, nccl-tests runs one more AllReduce (out-of-place and in-place) with new inputs and
compares every element of the output with the expected result, which it computes on the GPU; `#wrong` counts
the elements that differ. The timed iterations themselves are not checked; tests 01 and 02 check every
iteration. R2CC-AllReduce is also exercised with every message size in test 06. From
[logs/03.nccl_tests_compare_all.log](logs/03.nccl_tests_compare_all.log):

```
===== comparison (-b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum) =====
bytes        | baseline_healthy                 | balance_unhealthy                | r2cc_allreduce_unhealthy
268435456    | 3.36/2.76 (wrong 0/0)            | 2.24/2.31 (wrong 0/0)            | 2.37/2.50 (wrong 0/0)
1073741824   | 3.52/3.30 (wrong 0/0)            | 2.35/2.36 (wrong 0/0)            | 2.56/2.56 (wrong 0/0)
4294967296   | 3.53/3.53 (wrong 0/0)            | 2.35/2.34 (wrong 0/0)            | 2.56/2.56 (wrong 0/0)
(cells: busbw out-of-place/in-place GB/s, then #wrong out-of-place/in-place)
```

- All `#wrong` columns are 0: the checked AllReduce of every size and schedule matches the expected result.
- Bandwidth at 4 GiB: healthy 3.53 GB/s = three NICs at about 1.18 GB/s; R2CC-Balance 2.34–2.35 GB/s = two NICs;
  R2CC-AllReduce 2.56 GB/s. Relative to healthy (in-place, 4 GiB), 1.51 and 1.38 times the time, against 1.5 and
  1.375 in the model (section 1.3). R2CC-AllReduce gives 8.9% (out-of-place) and 9.4% (in-place) more bandwidth
  than Balance at 4 GiB and 8.9% and 8.5% at 1 GiB; the model predicts 9.1%. Below 4 GiB single passes vary more
  from run to run (in this run the healthy in-place pass at 1 GiB, 3.30 GB/s, and both healthy passes at 256 MiB).

### 3.4 Tests 04–06 — full nccl-tests sweeps

Each runs the standard `all_reduce_perf` sweep from 8 B to 4 GiB (`-b 8 -e 4G -f 2 -g 1 -c 1 -n 5 -w 2
-d float -o sum`, checked with `-c 1` as in 03) for one scenario and saves the complete output:

| Test | Scenario | busbw from 512 MiB to 4 GiB (out-of-place / in-place) | at 4 GiB |
|---|---|---|---|
| `04.nccl_tests_baseline_healthy.sh` | healthy, R2CC switched off (`R2CC_MODE=0`) | 3.49–3.53 / 3.44–3.53 GB/s | 3.53 / 3.53 GB/s |
| `05.nccl_tests_balance_unhealthy.sh` | R2CC-Balance, `mlx5_2` failed (`R2CC_MODE=2`) | 2.35–2.36 / 2.35–2.36 GB/s | 2.35 / 2.35 GB/s |
| `06.nccl_tests_r2cc_allreduce_unhealthy.sh` | R2CC-AllReduce, `mlx5_2` failed (`R2CC_MODE=3`) | 2.50–2.56 / 2.50–2.56 GB/s | 2.56 / 2.56 GB/s |

Notes: messages below `R2CC_AR_MIN_BYTES` (16 MiB) fall back to Balance, with one `falling back to Balance:
message below minimum size` warning per rank (printed into the first row of the table), so the small sizes in the
log of 06 are Balance numbers (8 MiB: 2.37/2.39 GB/s in 06, 2.42/2.39 GB/s in 05); from 16 MiB on, R2CC-AllReduce
is above Balance at every size in at least one of the two passes, and from 256 MiB on in both. Single sizes up to
256 MiB are occasionally slower in one of the two passes (in this run, for example, 04 at 64 MiB and 128 MiB, 05 at
64 MiB out-of-place and 128 MiB in-place, and 06 at 32 MiB and 128 MiB out-of-place); from 512 MiB on the results
vary by at most about 0.1 GB/s between runs. All three scripts take nccl-tests arguments and `NCCL_TEST_BIN` for other
collectives:

```bash
./06.nccl_tests_r2cc_allreduce_unhealthy.sh -b 1G -e 4G -f 4 -d half -o prod
NCCL_TEST_BIN=all_gather_perf ./05.nccl_tests_balance_unhealthy.sh -b 64M -e 1G -f 2 -d float
R2CC_AR_STAGE2_CHUNKS=8 ./06.nccl_tests_r2cc_allreduce_unhealthy.sh -b 4G -e 4G     # K = 8 instead of 4
```

`R2CC_AR_SCHEDULE` changes which stages of R2CC-AllReduce overlap (`06.nccl_tests_r2cc_allreduce_unhealthy.sh`
lists the values). The default with three servers, 1, runs the partial AllReduce concurrently with the AllReduce
of all ranks, as the model assumes; 2 runs it afterwards (the default with two servers), a case the model of
section 1.3 does not describe.

### 3.5 Test 07 — the pipeline depth K of R2CC-AllReduce against the model

`07.nccl_tests_r2cc_allreduce_k_sweep.sh` runs `all_reduce_perf` at 4 GiB (`-b 4G -e 4G -g 1 -c 1 -n 5 -w 2
-d float -o sum`) in nine configurations, with `mlx5_2` declared failed as in 05/06: healthy and R2CC-Balance,
R2CC-AllReduce with K = 1, 2, 4, 8 and 16 Stage-2 chunks (`R2CC_AR_STAGE2_CHUNKS`, with `R2CC_AR_SCHEDULE=1`),
then healthy and Balance again, which shows whether the testbed drifted during the sweep. The script divides every
time by the mean of the two healthy runs, separately for out-of-place and in-place, and prints it next to the
model of section 1.3. From [logs/07.nccl_tests_r2cc_allreduce_k_sweep.log](logs/07.nccl_tests_r2cc_allreduce_k_sweep.log):

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

- Every configuration is within 0.74% of the model and every `-c 1` check is 0. The healthy runs before and
  after the sweep differ by less than 0.1%, the Balance runs by at most 0.1%.
- With K = 1 the Broadcast of the tail starts only after the whole Reduce, and R2CC-AllReduce is slower than
  Balance (1.60 against 1.50). From K = 2 on it is faster. The time falls with K as the model predicts towards
  the K → ∞ value 1.30; at K = 16 it is 0.6–0.7% above the model, the cost of the many small chunks that the
  model leaves out.
- The default K = 4 gives 1.38 (1.376 and 1.377 in the two passes), the value used in sections 1.3, 3.2 and
  3.3.

## 4. CUDA graphs

`08.hot_repair_cuda_graph.sh` runs the hot repair of 01/02 with the AllReduce captured in a CUDA graph. It runs
`hot_repair/test_hot_repair_graph` twice, each with the real failure of 01/02 in iteration 3. The graph captured at
the start is replayed in every iteration, so the hot repair happens while it is replayed; after iteration 6 the
AllReduce is captured again, as R2CC-Balance in the first run and as R2CC-AllReduce (`R2CC_AR_AFTER_REPAIR=3`) in
the second, where one AllReduce outside the capture first creates its sub-communicators (`--eager`). Every
iteration gets new inputs, and every element of every AllReduce output, including the one outside the capture, is
checked on every rank as in 01/02 (section 3.1), outside the timed part. The table gives the time per iteration and
the re-capture overhead (the pre-failure rows are iteration 2 and iterations 4–6 of both runs, the re-captured rows
iterations 7–10); below it come the check and the failure evidence of each run. From
[logs/08.hot_repair_cuda_graph.log](logs/08.hot_repair_cuda_graph.log):

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

- **Every element of every AllReduce is correct in both runs**: in the iteration the failure hits, in the replays
  of the pre-failure graph over the backup connection, in the AllReduce outside the capture and in the replays of
  the re-captured graphs.

- **The graph captured before the failure keeps replaying without a new capture.** The hot repair only changes
  which connection the CPU proxy uses, and NCCL drives the proxy again on every replay, so the graph runs on
  with its kernels and buffer addresses unchanged. Its channel assignment is the one of the healthy cluster, so
  the backup connection carries the whole share of `mlx5_2` over `mlx5_0`, hence 4.07 s.
- **R2CC-Balance and R2CC-AllReduce are decided when a collective is enqueued**, so they apply to graphs captured
  after the repair, and the replayed graphs take about the same time as without graphs (01 and 02: 3.04 s and
  2.80 s). Capturing and instantiating the AllReduce again costs 1.7 ms for Balance; this capture is the first
  AllReduce call after the repair and also performs the repair's state exchange.
- **R2CC-AllReduce creates its two sub-communicators in its first eligible AllReduce, which cannot be captured.**
  Its re-capture overhead, 0.33 s, is almost entirely that one AllReduce outside the capture; a captured AllReduce
  before it falls back to Balance with a warning. Balance needs no such call: the AllReduce issued during the
  capture takes the repaired state.

## 5. Training through a NIC failure

`09.training_with_nic_failure.sh` trains GPT-2 (124M parameters) on WikiText-103 with PyTorch DDP on the six GPUs
through a real failure of node-1's `mlx5_2` and compares the training with failure-free training on upstream NCCL,
as in the paper's appendix on training quality. Every run (`training/train.py`) is 1000 optimizer updates with a
global batch of 48 sequences of 1024 tokens (AdamW, FP16 autocast with loss scaling). DDP puts the gradients of
all parameters into one bucket, so every update performs one AllReduce of 124,475,904 floats (475 MiB). The data
order, the initialization and the kernels are deterministic (fixed seed, `torch.use_deterministic_algorithms`, no
TF32), so two runs whose AllReduces return the same results are identical bit for bit. The script makes four runs
with the same seed (default 42):

| Run | Library | node-1's `mlx5_2` | Schedule after the hot repair |
|---|---|---|---|
| VNF | upstream NCCL 2.23.4 | healthy | – |
| NF | R2CC | healthy | – |
| BALF | R2CC | cut at update 400, down until the run ends | R2CC-Balance |
| ARF | R2CC | cut at update 400, down until the run ends | R2CC-AllReduce (`R2CC_AR_AFTER_REPAIR=3`) |

In BALF and ARF, rank 0 runs `nic/disconnect_nic1.sh` when update 400 starts. The library is not told which NIC
fails (no `R2CC_FAILED_*`); Balance and R2CC-AllReduce use the node and the NIC found by the hot repair. Every
run records:

- the training loss of every update (mean over the ranks), its duration and the MB that node-1's `mlx5_2`
  received during it (`train_log.csv`);
- for updates 400–408, the input and the output of the gradient AllReduce on every rank: the SHA-256 of the
  output and, computed after training, its error against the sum of the inputs of all ranks in FP64,
  E_rel = |output − FP64 sum| / |FP64 sum|;
- the SHA-256 of the parameters on every rank after updates 399, 402–406 and 1000;
- the test perplexity at the end (whole test split, windows of 1024 tokens with stride 512).

`training/compare.py` then prints, for every seed, each run against VNF: the first update in which `mlx5_2`
received less than a fifth of its usual traffic, up to which update the AllReduce outputs and the training loss
are identical to VNF's, the largest loss difference, the largest E_rel and the test perplexity. It ends with the test perplexity table of the paper
(mean over the seeds; Δ is the largest increase over VNF of the same seed) and these checks, all of which must
hold:

- NF is bit-identical to VNF: the loss of every update, the test loss, the AllReduce outputs and the parameters;
- in BALF and ARF, `mlx5_2` went down during the run;
- in BALF and ARF, the AllReduce that the failure hit returns the same output as in VNF, and the loss is
  identical to VNF's up to the failure;
- in every run, every captured AllReduce returns the same output on all ranks, with E_rel < 1e-6;
- in every run, the parameters are identical on all ranks at every check point.

From [logs/09.training_with_nic_failure.log](logs/09.training_with_nic_failure.log) (seed 42):

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

- **Without a failure, R2CC is identical to upstream NCCL bit for bit**: the same loss in all 1000 updates, the
  same AllReduce outputs and parameters, the same test perplexity to the last digit.
- **The hot repair returns the same bits as upstream NCCL.** The cut takes effect between the AllReduces of
  updates 400 and 401: `mlx5_2` received nothing during update 401, so its AllReduce starts on the dead port.
  R2CC detects the stalled transfers, moves them to the backup connection and completes the AllReduce, whose
  output equals VNF's; the loss is identical up to update 402 (the loss of an update is computed before its
  AllReduce). In `train_log.csv`, update 401 takes about 1.4 s instead of 0.5 s, and the training continues
  without a restart, at 0.62 s per update with Balance and 0.59 s with R2CC-AllReduce (about 0.9 s for update
  402, whose AllReduce creates the sub-communicators of R2CC-AllReduce).
- **From update 402 on**, Balance and R2CC-AllReduce split every AllReduce differently over the remaining NICs,
  so the gradients are summed in a different order. Floating-point addition is not associative, so the outputs
  differ from VNF's in the last bits; all of them, VNF's included, are within E_rel 5.8e-8 of the FP64 sum, the
  size of FP32 rounding. The trainings then drift apart by rounding: the loss differs from VNF's by at most
  1.31e-3 (Balance) and 1.76e-3 (R2CC-AllReduce) nats per token, and the test perplexity by +0.006% and +0.023%.

The runs are deterministic: the four runs of this log are bit-identical to the seed-42 runs reported in the paper
(every loss, AllReduce output and parameter hash). The paper reports seeds 42, 43 and 44 (`SEEDS="42 43 44"`,
about 2.3 hours); for its runs `training/compare.py` prints the table of the paper, and all checks hold:

```
Condition                  Test PPL   Max. paired Δ*
NCCL 2.23.4, no failure      76.085   --
R2CC, no failure             76.085   0.000%
R2CC-Balance, failure        76.085   +0.008%
R2CC-AllReduce, failure      76.086   +0.023%
```

Over the three seeds, the loss differs from VNF's by at most 2.43e-3 nats per token (seed 44, R2CC-AllReduce),
and the change of the test perplexity has no consistent sign: seed 44 ends 0.014% (Balance) and 0.033%
(R2CC-AllReduce) below VNF.

The runs use an environment prepared once on the shared storage of the CloudLab project,
`/proj/softmeasure-PG0/r2cc_ae` (`AE_ROOT`), which every server of the experiment mounts:

- `venv/`: Python 3.8 with PyTorch 2.4.1 built from source (tag v2.4.1, CUDA 12.2, sm_70) with
  `USE_SYSTEM_NCCL=1`. This PyTorch has no NCCL of its own and loads the `libnccl.so.2` found on
  `LD_LIBRARY_PATH`: R2CC from `/mydata/R2CC/build/lib`, or upstream NCCL. `train.py` checks on every rank that
  the library it loaded is the one the run asks for and that it reports version 2.23.4.
- `data/`: WikiText-103 (`wikitext-103-raw-v1` from the Hugging Face hub, revision `b08601e`) tokenized with the
  GPT-2 BPE; `manifest.json` lists the number of tokens and the SHA-256 of each split.
- `nccl_vanilla/`: upstream NCCL 2.23.4-1, the commit R2CC is based on, built with plain `make`
  (`PROVENANCE.txt`).

The files of every run (`train_log.csv`, `summary.json`, `stdout.log`) are kept in `OUT` (default
`/mydata/r2cc_training/<date and time>`, printed at the start); `training/compare.py <OUT> <seeds>` prints the
tables again.

## 6. Directory layout

```
01.hot_repair_to_balance.sh                real disconnect -> hot repair -> R2CC-Balance
02.hot_repair_to_r2cc_allreduce.sh         real disconnect -> hot repair -> R2CC-AllReduce
03.nccl_tests_compare_all.sh               the three nccl-tests scenarios (04-06) + comparison table
04.nccl_tests_baseline_healthy.sh          healthy, R2CC switched off, all NICs
05.nccl_tests_balance_unhealthy.sh         R2CC-Balance with mlx5_2 failed
06.nccl_tests_r2cc_allreduce_unhealthy.sh  R2CC-AllReduce with mlx5_2 failed
07.nccl_tests_r2cc_allreduce_k_sweep.sh    R2CC-AllReduce with K = 1-16 Stage-2 chunks vs. the model
08.hot_repair_cuda_graph.sh                real disconnect with the AllReduce in a CUDA graph (replay, re-capture): times and checks
09.training_with_nic_failure.sh            GPT-2 training: upstream NCCL / R2CC, without and with a real disconnect
common.sh                                  shared settings: mpirun line, failure model, check_idle, NIC restore, nccl-tests runner, table
nodes.sh                                   the servers of the experiment (node-1 + REMOTE_HOSTS) and their addresses
hot_repair/                                test_hot_repair.cc + run_hot_repair.sh (01/02), test_hot_repair_graph.cc (08),
                                           test_common.h (the check both use), Makefile, binaries
training/                                  train.py (one training run of 09), compare.py (its table and checks)
nic/                                       SmartNIC helpers: disconnect_nic1.sh / connect_nic1.sh (OVS drop rule),
                                           shape_nics.sh (NIC rate limits), check_ip.sh, setup and OVS checks
xml/                                       NCCL topology dumper and the equal-speed topo.xml used via NCCL_TOPO_FILE
setup/                                     02.setup_network_and_nic.sh, the Phase 2 script of r7525_setup.md
tools/                                     kill.sh, sync.sh (rsync repo to the other nodes), stress_test.sh, build_nccl_tests.sh
logs/                                      terminal output of one run of each test (01-09) on three servers
r7525_setup.md                             how to set up the r7525 servers
```
