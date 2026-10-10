# R2CC: Reliable and Resilient Collective Communication

R<sup>2</sup>CCL is a fault-tolerant communication library that provides lossless, low-overhead failover by
exploiting multi-NIC hardware, so a network failure no longer terminates the whole training or inference job. The
design and evaluation are in our NSDI paper ([arXiv](https://arxiv.org/abs/2512.25059)).

<p align="center">
  <img width="80%" src="./fig/r2cc-timeline.png"><br/>
  <b>R2CC (with failure) vs. Ideal (failure-free)</b>
</p>

Under a NIC failure, R2CC adds less than 1.1% overhead to training and less than 3% to inference, up to 92% and 98%
less than existing fault-tolerant systems.

## Features

- **Lossless in-flight failover.** When a NIC or link fails during a collective, R2CC moves the transfer to a backup
  NIC and resumes from the last confirmed step, so no data is lost and the job keeps running.
- **Balanced traffic (R2CC-Balance).** The failed NIC's traffic is spread over all remaining NICs instead of piling
  onto one.
- **Asymmetric AllReduce (R2CC-AllReduce).** The degraded server takes a smaller share of the AllReduce, so it no
  longer slows the whole ring.
- **CUDA Graphs.** Captured graphs keep replaying through a failure, and a re-capture picks up the optimized
  schedule.

<p align="center">
  <img width="70%" src="./fig/r2cc-allreduce-stages.png"><br/>
  <b>R2CC-AllReduce reduces the degraded node's load from 2D to 7/4D</b>
</p>

## Performance

<p align="center">
  <img width="90%" src="./fig/megatron-throughput.png"><br/>
  <b>Megatron training throughput under NIC failures</b>
</p>

## Demo

https://github.com/user-attachments/assets/8511cbf4-843a-4399-a742-d986eac55eb9

## Reproduce

[examples/cloudlab_r7525](./examples/cloudlab_r7525/README.md) runs R2CC through real NIC failures on three CloudLab
r7525 servers, with the scripts, the expected results and the reference logs of each test.

## Citation
```
@article{wang2025reliable,
  title={Reliable and Resilient Collective Communication Library for LLM Training and Serving},
  author={Wang, Wei and Yu, Nengneng and Xiong, Sixian and Liu, Zaoxing},
  journal={arXiv preprint arXiv:2512.25059},
  year={2025}
}
```
