# R2CC: Reliable and Resilient Collective Communication

**Keep GPU jobs running through network failures.**

NSDI '27 · [Paper](https://arxiv.org/abs/2512.25059) · [Demo](#demo) · [Getting started](#getting-started) ·
[Experiments](./examples/cloudlab_r7525/README.md)

A failed NIC or link should not force a job on healthy GPUs to restart. R<sup>2</sup>CCL repairs failed network
paths while collective operations are still running. When the GPUs are healthy and another network path remains, it
completes the interrupted operation without restarting the job, then redistributes the traffic and adapts the
AllReduce schedule to the remaining bandwidth.

<p align="center">
  <img width="80%" src="./fig/r2cc-timeline.png"><br/>
  <b>R2CC (with failure) vs. Ideal (failure-free)</b>
</p>

## Demo

https://github.com/user-attachments/assets/8511cbf4-843a-4399-a742-d986eac55eb9

On three CloudLab servers, tests [01](./examples/cloudlab_r7525/01.hot_repair_to_balance.sh) and
[02](./examples/cloudlab_r7525/02.hot_repair_to_r2cc_allreduce.sh) run ten 4 GiB AllReduces and cut the NIC `mlx5_2`
of node-1 in the middle of the third. That AllReduce still completes, and every element of all ten results is
correct. The data each server receives per AllReduce ([log](./examples/cloudlab_r7525/logs/rx_per_node.log)) shows
what the two schedules do after the cut.

| MB received per AllReduce | node-1 (lost `mlx5_2`) | node-2 | node-3 |
|---|---|---|---|
| Healthy | 7032 | 7032 | 7032 |
| R2CC-Balance after the cut | 7078 | 7078 | 7078 |
| R2CC-AllReduce after the cut | 6150 | 8230 | 8218 |

R2CC-Balance keeps node-1's load and spreads it over its two remaining NICs. R2CC-AllReduce lowers node-1's load by
13% and moves that share to the healthy servers.

## Results

The paper evaluates R2CC on four servers, each with eight H100 GPUs and eight 400 Gb/s InfiniBand NICs, with one
NIC failed, which removes 12.5% of that server's bandwidth.

- **Overhead against a failure-free run.** Less than 1.1% for training and less than 3% for inference.
- **Against existing fault-tolerant systems.** R2CC reduces the failure-induced overhead by up to 92% for training
  and up to 98% for inference.

<p align="center">
  <img width="90%" src="./fig/megatron-throughput.png"><br/>
  <b>Megatron training throughput under NIC failures</b>
</p>

## How it works

R2CC handles a failure in three steps, each building on the previous one.

1. **Finish the interrupted operation (HotRepair).** GPU buffers are registered with several NICs in advance. When
   a path fails, the two endpoints agree on the last chunk that was safely delivered, and only the data after it is
   resent over a backup connection, so the running collective completes with correct results instead of starting
   over.
2. **Rebalance the traffic (R2CC-Balance).** The backup NIC now carries extra traffic and becomes the bottleneck.
   R2CC spreads the failed NIC's share over all remaining NICs of the server.
3. **Adapt AllReduce (R2CC-AllReduce).** In a ring AllReduce every server moves the same amount of data, so the
   server that lost bandwidth sets the pace. R2CC-AllReduce reduces the communication load on that server with a
   partial AllReduce among the healthy servers and a broadcast of its result.

Collectives captured in CUDA Graphs keep replaying through a failure, and a re-capture picks up the optimized
schedule.

<p align="center">
  <img width="70%" src="./fig/r2cc-allreduce-stages.png"><br/>
  <b>R2CC-AllReduce reduces the degraded node's load from 2D to 7/4D</b>
</p>

## Getting started

```shell
git clone https://github.com/r2cc-project/R-2CCL.git
cd R-2CCL
make -j    # builds build/lib/libnccl.so
```

Programs that load NCCL dynamically pick up R2CC through `LD_LIBRARY_PATH`. Failover needs no setting, and the
collectives after a repair use R2CC-Balance, or R2CC-AllReduce with `R2CC_AR_AFTER_REPAIR=3`.

## Experiments

[examples/cloudlab_r7525](./examples/cloudlab_r7525/README.md) runs nine tests on three CloudLab r7525 servers, each
with a reference log. Tests 01, 02, 08 and 09 cut a real NIC in the middle of the run by dropping its traffic on the
BlueField SmartNIC, and tests 01–08 check every element of every collective result.

- **In-flight recovery** (01–02). The AllReduce that the failure hits completes with correct results.
- **Performance after a failure** (03–07). R2CC-Balance and R2CC-AllReduce bandwidth is within 1% of what the
  paper's formulas predict.
- **CUDA Graphs** (08). A graph captured before the failure keeps replaying correctly, and a re-captured graph uses
  the optimized schedule.
- **Training quality** (09). GPT-2 training through a NIC failure ends within 0.03% of the failure-free test
  perplexity.

## Citation
```
@article{wang2025reliable,
  title={Reliable and Resilient Collective Communication Library for LLM Training and Serving},
  author={Wang, Wei and Yu, Nengneng and Xiong, Sixian and Liu, Zaoxing},
  journal={arXiv preprint arXiv:2512.25059},
  year={2025}
}
```
