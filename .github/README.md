# Mooncake with FlashState for Kimi-K3.1-Pro on GB300

This is Mooncake v0.3.13.post1 with FlashState's store changes and three upstream fixes for GPU memory registered through
DMA-BUF. FlashState is a serving system for agentic workloads on hybrid models (attention plus linear-attention or
state-space layers). Its vLLM writes each finished request's hybrid state (a recurrent checkpoint and the attention KV it
depends on) to a Mooncake store; this fork's master evicts such a state whole, ranked by the prefill time it saves per
byte, instead of one object at a time.

**Results, the benchmark setups and installation: [FutureMLS-lab/FlashState](https://github.com/FutureMLS-lab/FlashState).**
The engine side is the `flashstate-k31pro-gb300` branch of
[Weili-0234/vllm](https://github.com/Weili-0234/vllm/tree/flashstate-k31pro-gb300), which also gives the Kimi-K3.1-Pro
launch commands. The eviction policy is described in
[`docs/source/design/store/unit-eviction-policy.md`](../docs/source/design/store/unit-eviction-policy.md).

## Options

| Option | Default | Effect |
| --- | --- | --- |
| `--eviction_policy=unit_gds` (master) | `lease` | evict whole recovery units (a state plus the pages it needs) that clients register on the admin port (`POST /api/v1/units`), by GreedyDual-Size over the prefill time each saves |
| `--unit_leaf_last`, `--unit_idle_demote_s S` (master, `unit_gds` only) | `true`, `60` | evict dead units first, then inner units, then leaves; a leaf unused for `S` seconds ranks as inner |
| `--unit_orphan_grace_s S` (master, `unit_gds` only) | `120` | evict objects that no unit owns or references first, once `S` seconds have passed since they were stored |
| `--unit_anchor_tier` (master, leaf-last only) | `false` | rank the aligned unit under a live tail with the leaves: a request that rewrites the end of the tail resumes there |
| `--unit_dead_headroom_ratio R` (master, leaf-last only) | `0` | once usage passes the high watermark minus `R`, evict dead units and expired orphans down to that level, never a live unit |
| `MC_GPU_COPY=pinned` (client) | unset | copy between GPU and host memory on a non-blocking stream through pinned buffers, so a put does not stall the engine's kernels |

With the defaults, this branch behaves as Mooncake v0.3.13.post1 with the three fixes below: the master refuses unit
updates, and every copy runs as before. The store protocol is unchanged, so stock clients and masters interoperate.

## GPU memory through DMA-BUF

GB300 trays have no GPU peer-memory module, so the transfer engine must register GPU memory through DMA-BUF
(`WITH_NVIDIA_PEERMEM=0`). This branch carries three upstream fixes for that path:

- #3538 by Liu Ke, with Feng Ren: export the requested range for dma-buf registration under CUDA VMM.
- #4998 by 0z5a: page-align the DMA-BUF export size for whole allocations.
- The pre-touch part of #4364 by cloudomate: skip pre-touch for DMA-BUF device memory.

The Kimi-K3.1-Pro GB300 image, `weili0234/flashstate:kimi-k31-cu134-266e88c-fs3f8611c-mc4592063-gb300` on Docker Hub
(also `weili0234/flashstate-vllm:v0.1.0-kimi-k31-gb300`), installs this branch's build of commit 4592063 (transfer
engine, store client and `mooncake_master`) in place of its base image's Mooncake.
