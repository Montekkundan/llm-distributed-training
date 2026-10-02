# Distributed training contract lab

This is a **CPU-first** student project for lectures 74–89 of *From random weights to your own LLM*. Its four original mathematical and ownership contract exercises use only the standard library:

1. **Data partitioning:** every sample has one owner; uneven local gradient sums must be reduced using the *global sample count*.
2. **Pipeline partitioning:** contiguous layer stages preserve forward and backward dependencies, and accumulated microbatch gradients equal a serial chain's gradients.
3. **Tensor partitioning:** a column split concatenates output features; a row split sums partial outputs. Both match a serial matrix-vector product.
4. **Context partitioning:** query and key/value blocks cover the sequence once; stable local softmax statistics merge to the exact causal-attention forward result.

An optional [PyTorch two-rank lab](src/distributed_lab/torch_distributed.py) adds **real Gloo collectives on two CPU processes** for a small column-then-row tensor-parallel FFN. Each rank owns half the hidden features. The lab compares the collective output, both weight-gradient shards, summed input gradient, and one sharded SGD step with serial autograd. This is a correctness check, not a speed or memory result. Supplementary reference modules now demonstrate ZeRO-style optimizer ownership, FSDP-style gather/gradient ownership, 1F1B and split-backward dependency clocks, GQA/MLA head partitioning, expert tensor parallelism, mesh groups, and atomic local multi-shard checkpoint/resume. They execute in one CPU process and do not implement production ZeRO/FSDP, Zero Bubble, DualPipe, or an NCCL training mesh. A small optional CUDA profiler/AMP audit is written to collect hardware evidence on a CUDA GPU, but its CUDA path has never been executed: only the CPU smoke path has run, and the checked-in tests establish CPU behavior only. The standard-library pipeline schedule is a sequential **fill-drain event trace**, and the context exercise tests blockwise softmax algebra without Ring Attention's communication. See [LESSON_MAP.md](LESSON_MAP.md) for the lesson-by-lesson scope.

A second [PyTorch DDP lab](src/distributed_lab/ddp_decoder.py) trains a one-block causal decoder on two CPU/Gloo ranks. It compares the average of two equal-sized local next-token losses, every synchronized parameter gradient, and one SGD update against a serial full-batch model. A causal-prefix test confirms that later tokens cannot change earlier logits. This is a numerical correctness exercise, not a GPU throughput result.

Two more [process-parallel checks](src/distributed_lab/process_parallel.py) use actual point-to-point Gloo communication. The pipeline check splits a small network across two processes, sends two microbatch activations forward and their gradients backward, and compares output, parameter gradients, and an SGD step to serial autograd. The context check assigns each process two consecutive query/key/value positions, exchanges key/value blocks, returns remote key/value gradients to their owner, and compares causal-attention output and all input gradients to serial attention. Its online softmax merge is blockwise, but this two-rank synchronous exchange is **not** Ring Attention's asynchronous, computation-overlapped schedule.

The [expert-parallel check](src/distributed_lab/expert_parallel.py) assigns two linear experts to each of two ranks. Five tokens request their top two experts; each expert accepts at most two assignments in token order, so two requests are dropped. In the default path rank 0 owns all five tokens and rank 1 owns none: rank 0 sends the remote experts' rows to rank 1 with point-to-point `dist.send`/`dist.recv`, combines weighted outputs, and sends output gradients back; the owner returns token-input gradients. This path is **not an all-to-all**, because there is a single source rank. `python3 -m distributed_lab.expert_parallel --all-to-all` runs the same routing with tokens on both ranks (token `t` on rank `t % 2`) and every row moved by `dist.all_to_all_single` with uneven splits, plus a first all-to-all that exchanges the row counts. It reports `rows_sent`, the matrix of rows each rank sends to each rank (`[[2, 3], [2, 1]]`). The capacity decision is replicated, not communicated: every rank holds all logits as constants and applies the same global token-major rule. In both paths forward values, expert/router/input gradients, and one update match a serial reference. Gate weights use a softmax over each token's original top two scores and are **not renormalized after capacity drops**. This is a fixed routing exercise, not DeepSeek-V3's gate, load-balancing strategy, or throughput profile.

## What is real and what is a reference

| Lectures | Modules | What runs |
| --- | --- | --- |
| 76, 79, 81, 84, 86 | `ddp_decoder.py`, `torch_distributed.py`, `process_parallel.py` (pipeline, context), `expert_parallel.py` | **Real two-rank Gloo runs** on two CPU processes. Each compares against serial autograd in float64 and matches to float64 rounding (the recorded runs show at most about 4e-16). The DDP lab also runs an unsynchronised negative control: rank 0's gradient without the all-reduce differs from the full-batch gradient by 0.19 (`local_only_gradient_max_error`). |
| 77, 78, 80, 82, 83, 85, 87, 88, 89 | `state.py`, `parallel_contracts.py`, `schedules.py`, `context.py`, `checkpoint.py` | **Single-process references.** One Python process emulates the partitioning or ledger; there is no inter-process communication, so no collective is exercised. 89 writes real local files but one process. |
| 74, 75 | `profiling.py` | **CPU smoke only.** The CUDA code path has never been executed and no GPU measurement is claimed. The `cuda` field of the report is `torch.version.cuda`, the CUDA version the wheel was built with, so it can be non-null on a CPU-only run. |

None of these is a speed, memory or scaling result. The Gloo runs are correctness checks of two tiny ranks.

## Cost formulas

[`costs.py`](src/distributed_lab/costs.py) is a pure-Python module of closed-form models with the equation and assumptions in each docstring: ring all-reduce, reduce-scatter and all-gather bytes (`2(N-1)/N * S` and `(N-1)/N * S`) with an alpha-beta time model, Megatron tensor-parallel collectives per block (2 forward and 2 backward all-reduces), the pipeline bubble `(p-1)/(m+p-1)`, ZeRO per-rank memory by stage, recompute overhead, and arithmetic intensity with the roofline ridge point `peak_flops / bandwidth`. Print worked examples with `PYTHONPATH=src python3 -m distributed_lab.costs`. The hardware numbers it prints are round illustrative values, not a named device. `tests/test_costs.py` checks every formula against hand-computed values, and checks `schedules.py` and `state.py` against the formulas:

- The 1F1B clock in `schedules.py` has makespan `(m+p-1)(F+B)` and therefore the bubble above, for every tested `p`, `m`, `F`, `B`. GPipe has the same bubble and differs only in saved activations (`m` per stage against `min(p-s, m)`).
- `state.zero_ledger` counts **18 bytes per parameter**: BF16 weights 2, FP32 gradient 4, FP32 master weight, Adam `m` and `v` 12. The ZeRO paper counts **16**: 16-bit gradient 2 plus the same 12. The whole difference is the gradient width, `2 * Psi` bytes in stages 0 and 1 and `2 * Psi / N` in stages 2 and 3, and `zero_stage_bytes_per_rank(..., grad_bytes=4)` reproduces the ledger exactly. Both exclude activations and buffers.
- `split_backward` in `schedules.py` has **no activation-memory cap by design**. It admits every forward early, so each stage's saved-input peak is `m` (`[m] * p`, for example `[6, 6, 6]` for `p=3, m=6`) against `[3, 2, 1]` for 1F1B, and for `m >= p` its makespan meets the lower bound `(p-1) + 3m` for unit F, B, W. It is the unconstrained end of the memory/bubble trade, not ZB-H1 or ZB-H2, which bound activation memory.
- `checkpoint_sequential` leaves its last segment un-checkpointed, so `profiling.py --segments k` re-runs `layers//k * (k-1)` of the `layers` layers, and `--segments 1` recomputes nothing. The test counts layer calls in PyTorch to confirm it.

## Run

Use Python 3.10 or newer from this directory; installation and a GPU are unnecessary:

```bash
PYTHONPATH=src python3 -m distributed_lab.checks
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m distributed_lab.costs
```

The first command prints four booleans and exits nonzero if any parity check fails. The second runs the invariant suite. The third prints the cost-formula examples (no PyTorch needed). Start at `src/distributed_lab/partitions.py`, then read `data.py`, `pipeline.py`, `tensor.py`, and `context.py`. Change a partition or reduction rule deliberately and use the tests to identify the broken contract.

For the real two-process lab, install the optional dependency in a virtual environment and run:

```bash
python3 -m pip install -e '.[torch]'
python3 -m distributed_lab.torch_distributed
python3 -m distributed_lab.ddp_decoder
python3 -m distributed_lab.process_parallel
python3 -m distributed_lab.expert_parallel
python3 -m distributed_lab.expert_parallel --all-to-all
python3 -m unittest discover -s tests -v
```

These labs use a fresh file rendezvous, Gloo, `float64` tensors, and two spawned CPU workers. On macOS they default to the `lo0` loopback interface unless `GLOO_SOCKET_IFNAME` is already set. They need permission to bind a local socket. Failures in either rank make the parent command fail. The standard-library suite still runs when PyTorch is absent; optional integration tests skip in that case. The DDP comparison assumes two equal-sized shards and a mean loss on each rank; uneven shards require weighting by sample/token count rather than naively averaging local means.

## Supplementary contracts

`state.py` checks a parameter/optimizer ownership ledger and sharded Adam update parity. `schedules.py` validates operation dependencies, stage resource use, and saved-activation lifetimes; its F/B/W policy is a reference list scheduler, not a reproduction of published Zero Bubble or DualPipe schedules. `parallel_contracts.py` compares head/hidden/context partitions and their autograd gradients with a serial function, then enumerates independent mesh groups. Its joint example composes dense DP/TP/CP ownership and a two-stage function in one process; it does not compose EP or physical pipeline communication. `checkpoint.py` publishes checksummed local shard files by same-filesystem directory rename and rejects incomplete/corrupt versions. It never overwrites a committed step: saving an existing step raises `ValueError` naming the directory and saying to use a new step or move or delete it first. The labs themselves use a fresh temporary rendezvous on every run, so re-running a command needs no cleanup. The tests compare exact six-step continuation with save at step three and resume, including Adam moments, RNG and data cursor. This local format is not an object-store or multi-host checkpoint service.

For GPU lessons, the following command requires CUDA and writes measured durations, memory statistics, and a Chrome profiler trace:

```bash
PYTHONPATH=src python3 -m distributed_lab.profiling --device cuda --precision bf16 --segments 2 --trace baseline.json
PYTHONPATH=src python3 -m distributed_lab.profiling --device cuda --precision bf16 --audit
```

The benchmark is a fixed synthetic MLP chain. It measures that workload, not complete LLM training or quality. Compare FP32/BF16/FP16 and checkpoint segment counts at identical shapes. CPU smoke mode supports `--device cpu --precision fp32`; its timings must not be reported as GPU measurements.

## Next implementation steps

Extend the two-rank labs to multi-GPU/NCCL with explicit device mapping and parity checks before measuring peak memory, throughput, and communication overlap on named hardware. Add real FSDP/ZeRO process groups, a fully overlapped context ring, physical expert-tensor groups, a learned router with load-balancing tests, and a measured 1F1B schedule. The reference identities and tests provide starting contracts for those implementations. The tensor-parallel FFN replicates the same batch on each rank and manually supplies the gradient of the reduced output to the local partial; it is not a general autograd-aware distributed operator. The DDP lab uses a fixed equal-shard batch rather than a distributed sampler, checkpointing, mixed precision, or gradient accumulation. A production implementation needs to handle those details and uneven token counts explicitly.

## Primary references

- [PyTorch DistributedDataParallel documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [distributed communication primitives](https://docs.pytorch.org/docs/stable/distributed/) for actual process groups and reductions.
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) for microbatch pipeline training.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) for tensor, pipeline, and data parallel composition.
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) for blockwise context-parallel attention.
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) for sparse top-2 expert routing and sharding.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) and [PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html) for sharded training state.
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) and the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) for advanced pipeline scheduling and DualPipe.

These references motivate future exercises. Passing this lab does **not** validate any of their full algorithms or performance claims.
