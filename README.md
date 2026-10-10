# R2CC: Reliable and Resilient Collective Communication

**Keep GPU jobs running through network failures.**

NSDI '27 · [Paper](https://arxiv.org/abs/2512.25059) · [Demo](#demo) · [Getting started](#getting-started) ·
[Experiments](./examples/cloudlab_r7525/README.md)

R2CC is an NCCL-based communication library that keeps distributed training and inference running through NIC and
link failures. It completes interrupted collectives over backup paths, rebalances traffic across surviving NICs,
and adapts AllReduce to the remaining bandwidth—all without restarting the job.

**Under a NIC failure, R2CC adds less than 1.1% training overhead and less than 3% inference overhead in the paper's H100 workloads.**

<p align="center">
  <img width="95%" src="./fig/r2cc-overview.png"><br/>
  <b>Detect failures, repair in-flight transfers, and adapt collective schedules at the communication layer.</b>
</p>

## Demo

https://github.com/user-attachments/assets/8511cbf4-843a-4399-a742-d986eac55eb9

<details>
<summary>Job progress: R2CC under failure vs. failure-free execution</summary>
<p align="center">
  <img width="80%" src="./fig/r2cc-timeline.png"><br/>
  <b>R2CC under failure compared with ideal, failure-free execution</b>
</p>
</details>

Tests [01](./examples/cloudlab_r7525/01.hot_repair_to_balance.sh) and
[02](./examples/cloudlab_r7525/02.hot_repair_to_r2cc_allreduce.sh) run ten 4 GiB AllReduces on three CloudLab
servers. During the third AllReduce, they block traffic on node-1's `mlx5_2` port. R2CC completes the interrupted
operation without a restart. In the reference runs, every output element passes validation on every rank in all
ten iterations.

The [reference receive-traffic measurements](./examples/cloudlab_r7525/logs/rx_per_node.log) show how the two
schedules respond to the failure.

| MiB received per AllReduce | node-1 (lost `mlx5_2`) | node-2 | node-3 |
|---|---|---|---|
| Healthy | 7032 | 7032 | 7032 |
| R2CC-Balance after the failure | 7078 | 7078 | 7078 |
| R2CC-AllReduce after the failure | 6150 | 8230 | 8218 |

R2CC-Balance distributes node-1's traffic across its two surviving NICs. R2CC-AllReduce reduces node-1's received
traffic by 13% relative to Balance, while the healthy servers perform additional communication over their
available paths.

## Results

The paper evaluates R2CC on an H100 testbed with four servers, each equipped with eight GPUs and eight 400 Gb/s
InfiniBand NICs. A single NIC failure removes 12.5% of the affected server's network bandwidth.

- **Near-baseline performance.** Less than 1.1% training overhead and less than 3% inference overhead relative to
  failure-free execution in the evaluated workloads.
- **Lower failure overhead.** Up to 92% less failure-induced overhead for training and 98% less for inference
  compared with the fault-tolerant systems evaluated in the paper.

<p align="center">
  <img width="90%" src="./fig/megatron-throughput.png"><br/>
  <b>Megatron-LM training throughput under a NIC failure</b><br/>
  With 32-GPU data parallelism, R2CC-AllReduce sustains 750,627 tokens/s versus 752,405 in the failure-free run
  (0.24% overhead). Vanilla NCCL aborts on the failure.
</p>

## How it works

1. **Detect and localize.** R2CC detects transport errors and stalled send completions, notifies the peer over an
   out-of-band channel, and coordinates failure information across ranks.
2. **Repair in flight (HotRepair).** GPU buffers are pre-registered with multiple NICs. The endpoints reconcile a
   protocol-aware replay boundary and retransmit only the unfinished suffix over a backup connection, completing
   the interrupted collective without restarting it. Recovery also supports NCCL's low-latency LL and LL128
   protocols.
3. **Adapt the schedule.** R2CC-Balance redistributes traffic across surviving NICs to avoid overloading a single
   backup path. R2CC-AllReduce further reduces communication on the bandwidth-constrained server.

### How R2CC-AllReduce reduces the bottleneck node's load

R2CC-AllReduce splits the tensor into two partitions. A global AllReduce processes the first across all ranks,
while a partial AllReduce processes the second among the healthy servers in parallel. The degraded server's
contribution is then combined with the second partition's partial result and broadcast to all ranks.

<p align="center">
  <img width="80%" src="./fig/r2cc-allreduce-stages.png"><br/>
  <b>Less communication on the degraded node</b><br/>
  In a four-node example with 25% bandwidth loss on one node, its communication load falls from 2D to 7D/4.
</p>

## Capabilities and integrations

| Capability / integration | Evidence | Tested configuration |
|---|---|---|
| nccl-tests | [Tests&nbsp;03&#8288;–&#8288;07](./examples/cloudlab_r7525/README.md#33-test-03--nccl-tests-correctness-and-a-side-by-side-table) | AllReduce correctness and performance, 8 B–4 GiB, in-place and out-of-place |
| CUDA Graphs | [Test&nbsp;08](./examples/cloudlab_r7525/README.md#36-test-08--cuda-graphs) | Correct replay through failure; re-capture enables optimized scheduling\* |
| PyTorch DDP | [Test&nbsp;09](./examples/cloudlab_r7525/README.md#37-test-09--training-through-a-nic-failure) | GPT-2 training through a persistent NIC failure, with gradient, parameter, loss, and perplexity comparisons |
| Megatron-LM | [Paper](https://arxiv.org/abs/2512.25059) | Data-parallel training (2.7B) and tensor-plus-pipeline-parallel training (13B) |
| vLLM | [Paper](https://arxiv.org/abs/2512.25059) | Inference under NIC failure, evaluated using TTFT and TPOT |
| AllGather, ReduceScatter, SendRecv | [Paper](https://arxiv.org/abs/2512.25059) | R2CC-Balance retains 83–90% of failure-free throughput for large messages |

\* Initialize R2CC-AllReduce with one eager AllReduce before capturing its optimized schedule for the first time.

## Getting started

**Requirements.** A multi-NIC RDMA environment with operational GPU endpoints and at least one surviving
inter-node path. R2CC is based on NCCL 2.23.4.

Build R2CC and add its library directory to the application's search path.

```shell
git clone https://github.com/r2cc-project/R-2CCL.git
cd R-2CCL
make -j
export LD_LIBRARY_PATH="$PWD/build/lib:${LD_LIBRARY_PATH:-}"
```

HotRepair is enabled by default. After recovery, R2CC uses R2CC-Balance. Set `R2CC_AR_AFTER_REPAIR=3` to use
R2CC-AllReduce.

**Library selection.** Configure the application to load R2CC's `libnccl.so`. Frameworks that bundle NCCL may need
additional configuration; verify the loaded library in `/proc/<pid>/maps`, as test 09 does.

For the CloudLab experiments, follow the [setup guide](./examples/cloudlab_r7525/r7525_setup.md), then run the
[scripts](./examples/cloudlab_r7525/README.md) from node-1, starting with `./01.hot_repair_to_balance.sh`.

## Experiments

The [CloudLab suite](./examples/cloudlab_r7525/README.md) provides nine experiments with reference logs covering
in-flight recovery, post-failure scheduling, CUDA Graphs, and training quality.

Tests 01, 02, 08, and 09 inject a NIC-port failure by blocking traffic on the BlueField SmartNIC during execution.
Tests 01, 02, and 08 verify every element of every AllReduce output; tests 03–07 use nccl-tests correctness checks
at each tested message size.

In test 07's reference 4 GiB K sweep, normalized completion times agree within 1% with the predictions of the
paper's formulas. The [experiment guide](./examples/cloudlab_r7525/README.md) explains how they are adapted to the
six-rank topology and finite pipeline depth.

## Citation
```
@article{wang2025reliable,
  title={Reliable and Resilient Collective Communication Library for LLM Training and Serving},
  author={Wang, Wei and Yu, Nengneng and Xiong, Sixian and Liu, Zaoxing},
  journal={arXiv preprint arXiv:2512.25059},
  year={2025}
}
```
