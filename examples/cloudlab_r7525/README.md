# R2CC experiments on CloudLab r7525

This document describes the six ready-to-run R2CC tests in this directory and the terminal output of one run of
each on three CloudLab `r7525` servers (`node-1`, `node-2`, `node-3`), kept in [`logs/`](logs/) for readers who
cannot run them. With every NIC rate-limited to the same speed, the tests check both sides of the paper: that
R2CC is correct (every result is verified, a real NIC failure is repaired in the middle of a collective, each NIC
carries the traffic the schedule assigns to it) and that R2CC-Balance and R2CC-AllReduce reach the bandwidth the
paper's model predicts. The document contains:

1. [Testbed](#1-testbed) — the servers as CloudLab provides them, why their bandwidth does not follow the model
   at line rate, the rate-limited configuration in which it does, and what the model predicts.
2. [Run the tests](#2-run-the-tests) — prerequisites, how the scripts behave, where the output goes.
3. [Results and analysis](#3-results-and-analysis) — for every test, the key lines of the saved log, what
   happened, how the NIC traffic changed and how the result compares with the model.
4. [CUDA graphs](#4-cuda-graphs) — the hot-repair test with its AllReduce captured in a CUDA graph.
5. [Directory layout](#5-directory-layout).

Bringing the machines up is described separately in [r7525_setup.md](r7525_setup.md).

| Test | What it shows | Reference log (three servers, NICs at 10 Gb/s) |
|---|---|---|
| `01.hot_repair_to_balance.sh` | real NIC failure during a 4 GiB AllReduce → hot repair → **R2CC-Balance** | [logs/01.hot_repair_to_balance.log](logs/01.hot_repair_to_balance.log) |
| `02.hot_repair_to_r2cc_allreduce.sh` | same failure → hot repair → **R2CC-AllReduce** | [logs/02.hot_repair_to_r2cc_allreduce.log](logs/02.hot_repair_to_r2cc_allreduce.log) |
| `03.nccl_tests_compare_all.sh` | nccl-tests, 256 MiB–4 GiB, NCCL healthy vs. Balance vs. R2CC-AllReduce, one table | [logs/03.nccl_tests_compare_all.log](logs/03.nccl_tests_compare_all.log) |
| `04.nccl_tests_baseline_healthy.sh` | full nccl-tests sweep (8 B–4 GiB), plain NCCL, all NICs healthy | [logs/04.nccl_tests_baseline_healthy.log](logs/04.nccl_tests_baseline_healthy.log) |
| `05.nccl_tests_balance_unhealthy.sh` | full sweep, R2CC-Balance, `mlx5_2` failed | [logs/05.nccl_tests_balance_unhealthy.log](logs/05.nccl_tests_balance_unhealthy.log) |
| `06.nccl_tests_r2cc_allreduce_unhealthy.sh` | full sweep, R2CC-AllReduce, `mlx5_2` failed | [logs/06.nccl_tests_r2cc_allreduce_unhealthy.log](logs/06.nccl_tests_r2cc_allreduce_unhealthy.log) |

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
- **At line rate the bandwidth does not follow the paper's model.** A ring enters a server through GPU0 and
  leaves it through GPU1: every byte passes NIC → GPU0 → GPU1 → NIC, through both GPUs' PCIe 3.0 x16 links (one
  GPU sends to the other at 11 GB/s here) and between the sockets. The NICs, 25 + 2 × 100 Gb/s, are not the
  narrowest part. A healthy ring already moves 8 GB/s per GPU and direction. R2CC-AllReduce runs the partial
  AllReduce of the healthy servers at the same time; it uses another NIC, but the two rings slow each other down
  on the GPU side and the overlap that the model assumes is lost. Measured at line rate (4 GiB AllReduce):
  healthy 5.7–8.0 GB/s; with `mlx5_2` failed, R2CC-Balance 5.6–5.7 GB/s and R2CC-AllReduce 3.2–5.4 GB/s,
  varying from run to run. The results stay correct; only the bandwidth is not representative.

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

For one degraded server among n, with a fraction X of its network bandwidth lost, the paper's model gives the
AllReduce time relative to the healthy one as 1 + X/(1−X) for R2CC-Balance and 1 + nX/(2(n−1)(1−X)) for
R2CC-AllReduce. Here n = 3 and X = 1/3 (one of three equal NICs fails):

| 4 GiB AllReduce | time relative to healthy, model | measured (nccl-tests, test 03) | measured (hot repair, tests 01/02) |
|---|---|---|---|
| NCCL, healthy | 1 | 2.03 s, busbw 3.53 GB/s | 2.03 s |
| R2CC-Balance, `mlx5_2` failed | 1.5 | 1.50 (3.04 s, 2.35 GB/s) | 1.50 (3.04 s) |
| R2CC-AllReduce, `mlx5_2` failed | 1.375 | 1.38 (2.79 s, 2.56 GB/s) | 1.38 (2.81 s) |

R2CC-AllReduce gives 9.1% more bandwidth than R2CC-Balance in the model and 8.9% in test 03. With two servers
both formulas give the same time: the partial AllReduce stays inside the single healthy server and R2CC-AllReduce
moves the same data over the network as Balance, so any gain needs three or more servers. The scripts still run
on two servers (`REMOTE_HOSTS=node-2`, section 2); the logs of the earlier two-server runs, without rate limits,
are in the history of this directory.

## 2. Run the tests

- Setup: [r7525_setup.md](r7525_setup.md) (CloudLab profile, SmartNIC firmware, network, `~/topo.xml`, NIC rate
  limits). For 03–06 build nccl-tests once with `tools/build_nccl_tests.sh`.
- Everything is run from `node-1` (`cd /mydata/R2CC/examples/cloudlab_r7525 && ./01.hot_repair_to_balance.sh`).
  The scripts use every server of the experiment: `nodes.sh` takes `node-2`, `node-3`, ... from `/etc/hosts`,
  and `REMOTE_HOSTS=node-2` restricts a run to two servers. `/mydata` is a per-node copy, so after any rebuild
  run `tools/sync.sh`.
- Limit the NICs first: `nic/shape_nics.sh 10` (`status` prints the limits, `off` removes them). Every test prints
  the limit of node-1's `mlx5_0` in a `[testbed]` line. Without the limits the tests still check correctness, but
  the bandwidth does not follow the model (section 1.1).
- One multi-node job at a time. Every script refuses to start while another one is running (`check_idle`
  in `common.sh`) and **every script first restores `mlx5_2` on the SmartNIC** (removes the OVS drop rule),
  so a killed run cannot leave the cluster degraded. `tools/kill.sh` stops leftover processes on all nodes.
- The scripts only print to the terminal; nothing is written to disk by default, so a local run can never
  overwrite the six reference logs in `logs/`. `SAVE_LOG=1 ./04.nccl_tests_baseline_healthy.sh` additionally
  saves the complete output to `logs/local/<NN>.<name>.log` (git-ignored; `LOG_DIR` changes the directory).

In every scenario the failed NIC is **node-1's `mlx5_2`**. Tests 01/02 really cut it on the BlueField
(`nic/disconnect_nic1.sh` installs an OVS drop rule for the port, `nic/connect_nic1.sh` removes it); tests
03–06 only declare it failed (`R2CC_FAILED_NODE=0`, `R2CC_FAILED_HCA=mlx5_2`), which gives the same degraded
topology without a disconnect.

## 3. Results and analysis

Each subsection quotes the key lines of the saved log of one run and explains what happened. All runs: three
servers, six GPUs, every NIC at 10 Gb/s. The `mlx5_*_RX` columns are the MB received by each of node-1's ports
during one iteration (`port_rcv_data`, including packet headers).

### 3.1 Test 01 — real NIC failure, hot repair, then R2CC-Balance

`hot_repair/test_hot_repair` runs 10 AllReduces of 4 GiB (float, sum) on the six GPUs and verifies every
result. Four seconds after the start, i.e. during iteration 3, the SmartNIC silently starts dropping all traffic
of node-1's `mlx5_2`. The library detects the stalled connection, live-migrates the in-flight transfers to the
backup connection and finishes the collective; from the next collective on it runs R2CC-Balance
(`R2CC_AR_AFTER_REPAIR=2`). The per-iteration table at the end shows the MB received by each of node-1's ports.
From [logs/01.hot_repair_to_balance.log](logs/01.hot_repair_to_balance.log):

```
[testbed] node-1 mlx5_0 egress ratelimit: 10.0 Gbps (nic/shape_nics.sh status shows all ports)
[Rank 0] Iter 2/10 END: OK (elapsed 2034 ms)
[Rank 0] Iter 3/10 START: allreduce 4.00 GiB
[Rank 0] NIC disconnect command completed.        <- mlx5_2 is now black-holed, iteration 3 is in flight
[Rank 0] Iter 3/10 END: OK (elapsed 3662 ms)      <- repaired mid-collective, result still verified
...
[Rank 0] IB RX per-iteration (MB, port_rcv_data *4B):
Iter   Time(ms)   mlx5_0_RX    mlx5_2_RX    mlx5_3_RX
1      2248       2406         2312         2312       <- iteration 1 includes connection setup
2      2034       2406         2312         2312       <- healthy: ~7.0 GB received, 2(n-1)/n x 4 GiB for n = 6
                                                          ranks plus headers, one third per NIC
3      3662       4330         467          2312       <- failure: mlx5_2 stops after 467 MB; the rest of its
                                                          share is migrated to the backup connection on mlx5_0
4      3042       3609         0            3468       <- R2CC-Balance: mlx5_2 unused, the same ~7 GB split
5      3046       3609         0            3468          over the two healthy NICs
...
10     3043       3609         0            3468
[Rank 0] TEST PASS: All allreduces completed and verified.
```

An iteration takes 2.03 s with three NICs and 3.04 s with two, 1.50 times as long, which is exactly the model's
1 + X/(1−X) for X = 1/3: the degraded server bounds the ring, and it now sends and receives the same data over
two NICs instead of three. The failover iteration costs about 1.6 s more than a healthy one. (Plain NCCL would
hang in iteration 3 until `NCCL_IB_TIMEOUT`/retry expire and then abort.)

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
2      2035       2406         2312         2312       <- healthy, as in test 01
3      3494       4131         657          2312       <- failure hits during iteration 3, the remainder of
                                                          mlx5_2's share is migrated to mlx5_0
4      3101       3129         0            3005       <- first R2CC-AllReduce: includes ncclCommSplit of the
                                                          two sub-communicators (one-off, ~0.3 s)
5      2797       3129         0            3005       <- steady state: node-1, the degraded server, receives
6      2796       3128         0            3005          ~6.1 GB per iteration instead of the ~7.1 GB of
...                                                       Balance: it takes part in the (1-X) AllReduce and
10     2830       3128         0            3005          receives the broadcast tail, but not the tail's AllReduce
[Rank 0] TEST PASS: All allreduces completed and verified.
```

An iteration now takes 2.81 s, 1.38 times the healthy time against 1.375 in the model, and 8% less than with
Balance (3.04 s). The traffic is the one the paper describes: node-1 receives 13% less than with Balance, which
is exactly (2(n−1)/n·(1−X) + X) / (2(n−1)/n) = 0.867 for n = 6 ranks; node-2 and node-3 absorb the tail
AllReduce on their otherwise idle `mlx5_2`.

### 3.3 Test 03 — nccl-tests: correctness and a side-by-side table

`03.nccl_tests_compare_all.sh` runs `all_reduce_perf` three times with identical arguments
(`-b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum`): plain NCCL on the healthy cluster
(`R2CC_MODE=0`), R2CC-Balance (`R2CC_MODE=2`) and R2CC-AllReduce (`R2CC_MODE=3`) with `mlx5_2` declared
failed, and prints one table. `-c 1` checks every result against the CPU reference (R2CC-AllReduce is also
exercised with every message size in test 06). From [logs/03.nccl_tests_compare_all.log](logs/03.nccl_tests_compare_all.log):

```
===== comparison (-b 256M -e 4G -f 4 -g 1 -c 1 -n 5 -w 2 -d float -o sum) =====
bytes        | baseline_healthy                 | balance_unhealthy                | r2cc_allreduce_unhealthy
268435456    | 3.53/3.52 (wrong 0/0)            | 2.35/2.35 (wrong 0/0)            | 2.56/2.56 (wrong 0/0)
1073741824   | 3.53/3.49 (wrong 0/0)            | 2.36/2.35 (wrong 0/0)            | 2.46/2.55 (wrong 0/0)
4294967296   | 3.53/3.53 (wrong 0/0)            | 2.35/2.35 (wrong 0/0)            | 2.56/2.56 (wrong 0/0)
(cells: busbw out-of-place/in-place GB/s, then #wrong out-of-place/in-place)
```

- All `#wrong` columns are 0: both R2CC schedules produce the same result as the CPU reference.
- Bandwidth: healthy 3.53 GB/s = three NICs at 1.18 GB/s; R2CC-Balance 2.35 GB/s = two NICs; R2CC-AllReduce
  2.56 GB/s. Relative to healthy, 1.50 and 1.38 times the time, against 1.5 and 1.375 in the model (section 1.3).
  R2CC-AllReduce gives 8.9% more bandwidth than Balance at 256 MiB and 4 GiB (the out-of-place pass at 1 GiB is
  a little slower); the model predicts 9.1%.

### 3.4 Tests 04–06 — full nccl-tests sweeps

Each runs the standard `all_reduce_perf` sweep from 8 B to 4 GiB (`-b 8 -e 4G -f 2 -g 1 -c 1 -n 5 -w 2
-d float -o sum`, results verified) for one scenario and saves the complete output:

| Test | Scenario | busbw from 256 MiB to 4 GiB (out-of-place / in-place) | at 4 GiB |
|---|---|---|---|
| `04.nccl_tests_baseline_healthy.sh` | plain NCCL, all three NICs (`R2CC_MODE=0`) | 3.40–3.53 / 3.51–3.53 GB/s | 3.53 / 3.53 GB/s |
| `05.nccl_tests_balance_unhealthy.sh` | R2CC-Balance, `mlx5_2` failed (`R2CC_MODE=2`) | 2.35 / 2.35–2.36 GB/s | 2.35 / 2.35 GB/s |
| `06.nccl_tests_r2cc_allreduce_unhealthy.sh` | R2CC-AllReduce, `mlx5_2` failed (`R2CC_MODE=3`) | 2.44–2.56 / 2.45–2.56 GB/s | 2.56 / 2.56 GB/s |

Notes: messages below `R2CC_AR_MIN_BYTES` (16 MiB) fall back to Balance, with one `falling back to Balance:
message below minimum size` warning per rank, so the small sizes in the log of 06 are Balance numbers (8 MiB:
2.37 GB/s in both 05 and 06); from 16 MiB on, R2CC-AllReduce is above Balance at every size. Single sizes in
the middle of a sweep (16–128 MiB) are occasionally slower in one of the two passes; from 256 MiB on the results
vary by at most about 0.1 GB/s between runs. All three scripts take nccl-tests arguments and `NCCL_TEST_BIN` for other
collectives:

```bash
./06.nccl_tests_r2cc_allreduce_unhealthy.sh -b 1G -e 4G -f 4 -d half -o prod
NCCL_TEST_BIN=all_gather_perf ./05.nccl_tests_balance_unhealthy.sh -b 64M -e 1G -f 2 -d float
R2CC_AR_STAGE2_CHUNKS=8 R2CC_AR_SCHEDULE=2 ./06.nccl_tests_r2cc_allreduce_unhealthy.sh -b 4G -e 4G
```

## 4. CUDA graphs

`hot_repair/test_hot_repair` can capture its AllReduce into a CUDA graph and replay the graph in every
iteration (every result is still verified). The options are environment variables, passed through by
`hot_repair/run_hot_repair.sh` and therefore by 01/02:

| Variable | Effect |
|---|---|
| `R2CC_TEST_GRAPH=1` | capture the AllReduce once at the start and replay it in every iteration |
| `R2CC_TEST_RECAPTURE_AT=<n>` | capture it again after iteration n, e.g. after the failure |
| `R2CC_TEST_EAGER_BEFORE_RECAPTURE=1` | run one AllReduce outside the capture before capturing again |
| `R2CC_TEST_NEWBUF_AT=<n>`, `R2CC_TEST_REGISTER=1` | from iteration n+1 on use a newly allocated buffer, registered with `ncclCommRegister` if requested |

```bash
R2CC_TEST_GRAPH=1 ./01.hot_repair_to_balance.sh                    # graph captured before the failure
R2CC_TEST_GRAPH=1 R2CC_TEST_RECAPTURE_AT=6 ./01.hot_repair_to_balance.sh
R2CC_TEST_GRAPH=1 R2CC_TEST_RECAPTURE_AT=6 R2CC_TEST_EAGER_BEFORE_RECAPTURE=1 ./02.hot_repair_to_r2cc_allreduce.sh
./03.nccl_tests_compare_all.sh -b 4G -e 4G -g 1 -c 1 -n 2 -w 2 -G 5   # nccl-tests replays from a graph
```

Measured on the same testbed (these runs are not part of `logs/`):

| 4 GiB AllReduce replayed from a CUDA graph | Time / iteration | node-1 RX MB (`mlx5_0` / `mlx5_2` / `mlx5_3`) |
|---|---|---|
| Captured before the failure, healthy | 2.03 s | 2406 / 2312 / 2312 |
| Same graph after `mlx5_2` fails, not captured again | 4.08 s | 4816 / 0 / 2312 |
| Captured again after the repair: R2CC-Balance | 3.04 s | 3609 / 0 / 3468 |
| Captured again after the repair: R2CC-AllReduce | 2.80 s | 3128 / 0 / 3022 |

- **Hot repair needs no new capture.** It only changes which connection the CPU proxy uses, and NCCL drives the
  proxy again on every replay, so the graph captured before the failure keeps replaying correctly with its
  kernels and buffer addresses unchanged. Its channel assignment is the one of the healthy cluster, so the backup
  connection carries the whole share of `mlx5_2` over `mlx5_0` (4.8 GB per iteration), hence 4.08 s.
- **R2CC-Balance and R2CC-AllReduce are decided when a collective is enqueued**, so they apply to graphs captured
  after the repair, and the replayed graphs run exactly as fast as without graphs (03 with `-G 5`: 3.53 / 2.35 /
  2.56 GB/s). Capturing and instantiating the AllReduce again takes at most 1.3 ms (6 graph nodes for Balance, 31
  for R2CC-AllReduce); the 1.3 ms is a Balance capture that also performs the repair's state exchange.
- **R2CC-AllReduce creates its two sub-communicators in its first eligible AllReduce, which cannot be captured.**
  Run one AllReduce outside the capture first (`R2CC_TEST_EAGER_BEFORE_RECAPTURE=1`; here 3.08 s, one normal
  iteration plus the creation of the sub-communicators); a captured AllReduce before that falls back to Balance
  with a warning. Balance needs no such call: the AllReduce issued during the capture takes the repaired state.

## 5. Directory layout

```
01.hot_repair_to_balance.sh                real disconnect -> hot repair -> R2CC-Balance
02.hot_repair_to_r2cc_allreduce.sh         real disconnect -> hot repair -> R2CC-AllReduce
03.nccl_tests_compare_all.sh               the three nccl-tests scenarios (04-06) + comparison table
04.nccl_tests_baseline_healthy.sh          plain NCCL, all NICs
05.nccl_tests_balance_unhealthy.sh         R2CC-Balance with mlx5_2 failed
06.nccl_tests_r2cc_allreduce_unhealthy.sh  R2CC-AllReduce with mlx5_2 failed
common.sh                                  shared settings: mpirun line, failure model, check_idle, NIC restore, nccl-tests runner, table
nodes.sh                                   the servers of the experiment (node-1 + REMOTE_HOSTS) and their addresses
hot_repair/                                test_hot_repair.cc (+ Makefile, binary), run_hot_repair.sh (driver of 01/02)
nic/                                       SmartNIC helpers: disconnect_nic1.sh / connect_nic1.sh (OVS drop rule),
                                           shape_nics.sh (NIC rate limits), check_ip.sh, setup and OVS checks
xml/                                       NCCL topology dumper and the equal-speed topo.xml used via NCCL_TOPO_FILE
setup/                                     02.setup_network_and_nic.sh, the Phase 2 script of r7525_setup.md
tools/                                     kill.sh, sync.sh (rsync repo to the other nodes), stress_test.sh, build_nccl_tests.sh
logs/                                      terminal output of one run of each test (01-06) on three servers
r7525_setup.md                             how to set up the r7525 servers
```
