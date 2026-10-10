# NSDI Artifact Evaluation

For the artifact evaluation, we provide three CloudLab r7525 servers with everything installed, available throughout
the artifact evaluation. The SSH key and the login instructions are on HotCRP. If you run into any problems,
we are always available on HotCRP.

Note that the testbed differs from the H100 servers in the paper. Our reproducible claim therefore rests on
trends consistent with the H100 results in the paper and on measured performance within 1% of what the paper's formulas predict under
this testbed's settings. The two experiments added during shepherding, CUDA Graphs and training quality, are fully
reproducible.

## Experiments

| Tests | What they show | Time |
|---|---|---|
| 01–02 | In-flight recovery is fast, lossless and correct | ~40 s each |
| 03–07 | Scheduling optimizations follow the paper's trends, within 1% of its formulas | ~16 min |
| 08 | CUDA Graphs support (added during shepherding) | ~1.5 min |
| 09 | Training quality under a NIC failure (added during shepherding) | ~45 min |

Tests 01–08 all check the correctness of the collective results, both on the healthy network and with a failed
NIC.

The [experiment README](./examples/cloudlab_r7525/README.md) gives the detailed experiment description, run
commands, expected results, reference logs and analysis of each experiment.

## Changes in the shepherding version

At the reviewers' request, the current version adds two experiments. The evaluation section summarizes the
training-quality experiment and the appendix gives its details. The CUDA Graphs compatibility experiment is in the Discussion
appendix.
