# R2CC experiments on CloudLab r7525

This document describes the six ready-to-run R2CC tests in this directory for two CloudLab `r7525` servers
(`node-1`, `node-2`) and the terminal output of one run of each, kept in [`logs/`](logs/) for readers who
cannot run them. It contains:

1. [Testbed](#1-testbed) — the two machines, why their topology is unusual and what that means for the numbers.
2. [Run the tests](#2-run-the-tests) — prerequisites, how the scripts behave, where the output goes.
3. [Example results and analysis](#3-example-results-and-analysis) — for every test, the key lines of the saved
   log with annotations of what happened and how the NIC traffic changed.
4. [Directory layout](#4-directory-layout).

Bringing the machines up is described separately in [r7525_setup.md](r7525_setup.md).

| Test | What it shows | Reference log (one run on our testbed) |
|---|---|---|
| `01.hot_repair_to_balance.sh` | real NIC failure during a 4 GiB AllReduce → hot repair → **R2CC-Balance** | [logs/01.hot_repair_to_balance.log](logs/01.hot_repair_to_balance.log) |
| `02.hot_repair_to_r2cc_allreduce.sh` | same failure → hot repair → **R2CC-AllReduce** | [logs/02.hot_repair_to_r2cc_allreduce.log](logs/02.hot_repair_to_r2cc_allreduce.log) |
| `03.nccl_tests_compare_all.sh` | nccl-tests, 256 MiB–4 GiB, NCCL healthy vs. Balance vs. R2CC-AllReduce, one table | [logs/03.nccl_tests_compare_all.log](logs/03.nccl_tests_compare_all.log) |
| `04.nccl_tests_baseline_healthy.sh` | full nccl-tests sweep (8 B–4 GiB), plain NCCL, all NICs healthy | [logs/04.nccl_tests_baseline_healthy.log](logs/04.nccl_tests_baseline_healthy.log) |
| `05.nccl_tests_balance_unhealthy.sh` | full sweep, R2CC-Balance, `mlx5_2` failed | [logs/05.nccl_tests_balance_unhealthy.log](logs/05.nccl_tests_balance_unhealthy.log) |
| `06.nccl_tests_r2cc_allreduce_unhealthy.sh` | full sweep, R2CC-AllReduce, `mlx5_2` failed | [logs/06.nccl_tests_r2cc_allreduce_unhealthy.log](logs/06.nccl_tests_r2cc_allreduce_unhealthy.log) |

## 1. Testbed

The topology of these machines is unusual; read this before looking at any number.

```
node-1 ──┬── GPU0 (V100S, NUMA 0)          node-2: identical
         ├── GPU1 (V100S, NUMA 1)          no NVLink; GPU0 <-> GPU1 traffic crosses the CPU interconnect
         ├── mlx5_0   25 Gb/s  (also carries the bootstrap / MPI interface eno33np0)
         ├── mlx5_2  100 Gb/s  BlueField port  <- the NIC that "fails" in every scenario
         └── mlx5_3  100 Gb/s  BlueField port
```

- The two GPUs of a node sit on **different NUMA nodes and have no NVLink**, so every intra-node hop of a ring
  goes through PCIe and the CPU interconnect. The three usable ports are **one 25G and two 100G**.
- On this machine a single 100G port is more than the GPUs can drive; **plain NCCL would simply pick one 100G
  NIC** and there would be nothing to fail over between. `xml/cloudlab_dump_topo.sh` therefore writes a
  topology file (`~/topo.xml`, passed with `NCCL_TOPO_FILE`) in which **every NIC is declared with the same
  speed**. NCCL then builds one ring per NIC and splits the data equally over the three, i.e. every NIC is
  effectively used as a 25G NIC and the aggregate is capped by the slowest one. This is what makes the
  "one of three NICs fails" experiments of the paper possible on two servers.
- Consequently these tests **validate correctness and behaviour** (failover, traffic re-distribution, the
  two-stage R2CC-AllReduce). The absolute and relative bandwidth numbers are **not representative** of a
  normal multi-NIC GPU server: they are bounded by the 25G port, by the CPU interconnect, by PCIe V100S
  without NVLink, and they vary noticeably from run to run.

## 2. Run the tests

- Setup: [r7525_setup.md](r7525_setup.md) (CloudLab profile, SmartNIC firmware, network, `~/topo.xml`).
  For 03–06 build nccl-tests once with `tools/build_nccl_tests.sh`.
- Everything is run from `node-1` (`cd /mydata/R2CC/examples/cloudlab_r7525 && ./01.hot_repair_to_balance.sh`).
  `/mydata` is a per-node copy, so after any rebuild run `tools/sync.sh`.
- One multi-node job at a time. Every script refuses to start while another one is running (`check_idle`
  in `common.sh`) and **every script first restores `mlx5_2` on the SmartNIC** (removes the OVS drop rule),
  so a killed run cannot leave the cluster degraded. `tools/kill.sh` stops leftover processes on both nodes.
- The scripts only print to the terminal; nothing is written to disk by default, so a local run can never
  overwrite the six reference logs in `logs/`. `SAVE_LOG=1 ./04.nccl_tests_baseline_healthy.sh` additionally
  saves the complete output to `logs/local/<NN>.<name>.log` (git-ignored; `LOG_DIR` changes the directory).

In every scenario the failed NIC is **node-1's `mlx5_2`**. Tests 01/02 really cut it on the BlueField
(`nic/disconnect_nic1.sh` installs an OVS drop rule for the port, `nic/connect_nic1.sh` removes it); tests
03–06 only declare it failed (`R2CC_FAILED_NODE=0`, `R2CC_FAILED_HCA=mlx5_2`), which gives the same degraded
topology without a disconnect.

## 3. Example results and analysis

Each subsection quotes the key lines of the saved log of one run and explains what happened. In every
scenario the failed NIC is node-1's `mlx5_2`; the `mlx5_*_RX` columns are the bytes received by each of
node-1's ports during one iteration.

### 3.1 Test 01 — real NIC failure, hot repair, then R2CC-Balance

`hot_repair/test_hot_repair` runs 10 AllReduces of 4 GiB (float, sum) on the 4 GPUs and verifies every
result. Four seconds after the start, i.e. during iteration 5/6, the SmartNIC silently starts dropping all
traffic of node-1's `mlx5_2`. The library detects the stalled connection, live-migrates the in-flight
transfers to the backup connection and finishes the collective; from the next collective on it runs
R2CC-Balance (`R2CC_AR_AFTER_REPAIR=2`). The per-iteration table at the end shows the bytes received by each
of node-1's ports (`port_rcv_data`). From [logs/01.hot_repair_to_balance.log](logs/01.hot_repair_to_balance.log):

```
[Rank 0] Arming NIC disconnect at program start (delay 4000 ms) using: ./nic/disconnect_nic1.sh
[Rank 0] Iter 5/10 END: OK (elapsed 843 ms)
[Rank 0] Iter 6/10 START: allreduce 4.00 GiB
[Rank 0] NIC disconnect command completed.        <- mlx5_2 is now black-holed, iteration 6 is in flight
[Rank 0] Iter 6/10 END: OK (elapsed 1313 ms)      <- repaired mid-collective, result still verified
...
[Rank 0] IB RX per-iteration (MB, port_rcv_data *4B):
Iter   Time(ms)   mlx5_0_RX    mlx5_2_RX    mlx5_3_RX
1      885        2165         2136         2166       <- healthy: ~6.4 GB received per iteration (2(n-1)/n x 4 GiB),
2      816        2165         2239         2293          split ~1/3 per NIC because all three are declared equal
3      842        2165         2319         2361
4      841        2165         2338         2355
5      843        2165         2348         2337
6      1313       3203         1251         2279       <- failure: mlx5_2 stops after 1251 MB; the rest of its share
                                                         is migrated to the backup connection on mlx5_0
7      1136       3249         0            3174       <- R2CC-Balance: mlx5_2 unused, the same 6.4 GB now split
8      1161       3249         0            3178          over the two healthy NICs (~3.2 GB each)
9      1128       3249         0            3149
10     1130       3249         0            3180
[Rank 0] TEST PASS: All allreduces completed and verified.
```

Per-iteration time goes from ~0.85 s (3 NICs) to ~1.15 s (2 NICs) instead of failing, and the failover
iteration itself costs about half a second of extra time. (Plain NCCL would hang in iteration
5 until `NCCL_IB_TIMEOUT`/retry expire and then abort.)

### 3.2 Test 02 — real NIC failure, hot repair, then R2CC-AllReduce

Identical run, but after the repair the library switches to R2CC-AllReduce (`R2CC_AR_AFTER_REPAIR=3`).
With X = failed NICs / NICs per node = 1/3, every AllReduce becomes:

- **Stage 1** — AllReduce of the first (1−X) of the buffer on all 4 ranks (over the healthy NICs), and the
  partial AllReduce of the last X on the healthy node's sub-communicator (node-2's two GPUs).
- **Stage 2** — the tail X is cut into K = 4 chunks (`R2CC_AR_STAGE2_CHUNKS`); for each chunk a Reduce of the
  degraded node's data onto a helper rank of the healthy node (whose input is the partial result of Stage 1)
  is pipelined with a Broadcast of the finished chunk back to all ranks.

The two sub-communicators are created lazily, on the first AllReduce that runs in this mode. From
[logs/02.hot_repair_to_r2cc_allreduce.log](logs/02.hot_repair_to_r2cc_allreduce.log):

```
Iter   Time(ms)   mlx5_0_RX    mlx5_2_RX    mlx5_3_RX
1      1100       2166         2083         2094       <- healthy, as in test 01
...
5      1673       3527         796          2094       <- failure hits during iteration 5 (796 of ~2100 MB had
                                                         arrived on mlx5_2), the remainder is migrated
6      1897       2889         0            2795       <- first R2CC-AllReduce: includes ncclCommSplit of the two
                                                         sub-communicators (one-off cost)
7      1320       2889         0            2792       <- steady state: node-1 (the degraded server) now receives
8      1404       2889         0            2794          ~5.7 GB per iteration instead of the 6.4 GB of Balance:
9      1616       2889         0            2796          it takes part only in the (1-X) AllReduce and in the
10     1308       2889         0            2791          Reduce/Broadcast of the tail, not in the tail's AllReduce
[Rank 0] TEST PASS: All allreduces completed and verified.
```

The traffic pattern is the one described in the paper: the degraded server moves less data over its two
remaining NICs, the healthy server absorbs the tail AllReduce over all of its NICs. Per-iteration times are
in the same range as Balance here — see the next section for why the two are expected to be equal on two
servers and why the measured numbers should not be over-interpreted.

### 3.3 Test 03 — nccl-tests: correctness and a side-by-side table

`03.nccl_tests_compare_all.sh` runs `all_reduce_perf` three times with identical arguments
(`-b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum`): plain NCCL on the healthy cluster
(`R2CC_MODE=0`), R2CC-Balance (`R2CC_MODE=2`) and R2CC-AllReduce (`R2CC_MODE=3`) with `mlx5_2` declared
failed, and prints one table. `-c 1` checks every result against the CPU reference, so **the point of this
test is the `#wrong` columns being 0** for both R2CC strategies (R2CC-AllReduce is also exercised with every
data type / reduction op and message size in test 06). From [logs/03.nccl_tests_compare_all.log](logs/03.nccl_tests_compare_all.log):

```
===== comparison (-b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum) =====
bytes        | baseline_healthy                 | balance_unhealthy                | r2cc_allreduce_unhealthy        
268435456    | 4.93/7.67 (wrong 0/0)            | 5.04/4.55 (wrong 0/0)            | 3.51/4.56 (wrong 0/0)           
1073741824   | 7.74/7.67 (wrong 0/0)            | 4.83/4.99 (wrong 0/0)            | 3.96/4.47 (wrong 0/0)           
4294967296   | 7.68/7.69 (wrong 0/0)            | 5.18/5.18 (wrong 0/0)            | 4.25/4.38 (wrong 0/0)           
(cells: busbw out-of-place/in-place GB/s, then #wrong out-of-place/in-place)
```

How to read the bandwidth columns:

- **R2CC-AllReduce is, in theory, exactly as fast as Balance on two servers.** The paper's gain over Balance
  comes from the *healthy* servers doing the tail AllReduce among themselves while the degraded server only
  handles (1−X) of the data; with n = 2 servers the "healthy servers" are a single machine, the tail
  AllReduce is intra-node, and what is left on the wire is the same amount of data Balance moves over the
  same two NICs. Any gain needs ≥ 3 servers (the degraded server's NICs stop being the bottleneck for the
  other n−1).
- **What the numbers on this machine do show** is the cost of the machine itself: no NVLink, the tail
  AllReduce and every intra-node hop crossing the CPU interconnect, the 25G port bounding every NIC, plus the
  extra kernel launches of a two-stage algorithm on a 4 GPU job. In the saved run R2CC-AllReduce reaches about
  80 % of Balance at 1–4 GiB; in other runs on the same machines the two were within noise of each other, and
  plain NCCL on the healthy cluster varied between 4.5 and 7.7 GB/s at 4 GiB. Treat the table as a correctness
  result with an indicative ordering, not as a performance claim.

### 3.4 Tests 04–06 — full nccl-tests sweeps

Each runs the standard `all_reduce_perf` sweep from 8 B to 4 GiB (`-b 8 -e 4G -f 2 -g 1 -c 1 -n 5 -w 2
-d float -o sum`, results verified) for one scenario and saves the complete output:

| Test | Scenario | 4 GiB busbw in the saved run (out-of-place / in-place) |
|---|---|---|
| `04.nccl_tests_baseline_healthy.sh` | plain NCCL, all three NICs (`R2CC_MODE=0`) | 5.48 / 5.35 GB/s |
| `05.nccl_tests_balance_unhealthy.sh` | R2CC-Balance, `mlx5_2` failed (`R2CC_MODE=2`) | 5.68 / 5.64 GB/s |
| `06.nccl_tests_r2cc_allreduce_unhealthy.sh` | R2CC-AllReduce, `mlx5_2` failed (`R2CC_MODE=3`) | 4.82 / 4.81 GB/s |

Notes on 06: messages below `R2CC_AR_MIN_BYTES` (16 MiB) fall back to Balance, so the small sizes in its log
are Balance numbers; the first eligible size pays the one-off sub-communicator creation inside its warm-up.
All three scripts take nccl-tests arguments and `NCCL_TEST_BIN` for other collectives:

```bash
./06.nccl_tests_r2cc_allreduce_unhealthy.sh -b 1G -e 4G -f 4 -d half -o prod
NCCL_TEST_BIN=all_gather_perf ./05.nccl_tests_balance_unhealthy.sh -b 64M -e 1G -f 2 -d float
R2CC_AR_STAGE2_CHUNKS=8 R2CC_AR_SCHEDULE=1 ./06.nccl_tests_r2cc_allreduce_unhealthy.sh -b 4G -e 4G
```

## 4. Directory layout

```
01.hot_repair_to_balance.sh                real disconnect -> hot repair -> R2CC-Balance
02.hot_repair_to_r2cc_allreduce.sh         real disconnect -> hot repair -> R2CC-AllReduce
03.nccl_tests_compare_all.sh               the three nccl-tests scenarios (04-06) + comparison table
04.nccl_tests_baseline_healthy.sh          plain NCCL, all NICs
05.nccl_tests_balance_unhealthy.sh         R2CC-Balance with mlx5_2 failed
06.nccl_tests_r2cc_allreduce_unhealthy.sh  R2CC-AllReduce with mlx5_2 failed
common.sh                                  shared settings: mpirun line, failure model, check_idle, NIC restore, nccl-tests runner, table
hot_repair/                                test_hot_repair.cc (+ Makefile, binary), run_hot_repair.sh (driver of 01/02)
nic/                                       SmartNIC helpers: disconnect_nic1.sh / connect_nic1.sh (OVS drop rule), setup, checks
xml/                                       NCCL topology dumper and the equal-speed topo.xml used via NCCL_TOPO_FILE
tools/                                     kill.sh, sync.sh (rsync repo to node-2), stress_test.sh, build_nccl_tests.sh
logs/                                      terminal output of one run of each test (01-06)
r7525_setup.md                             how to set up the two r7525 nodes
```
