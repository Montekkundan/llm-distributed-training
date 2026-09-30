# Distributed training contract lab

This is a **CPU-first** student project for lectures 74–89 of *From random weights to your own LLM*. Its four original mathematical and ownership contract exercises use only the standard library:

1. **Data partitioning:** every sample has one owner; uneven local gradient sums must be reduced using the *global sample count*.
2. **Pipeline partitioning:** contiguous layer stages preserve forward and backward dependencies, and accumulated microbatch gradients equal a serial chain's gradients.
3. **Tensor partitioning:** a column split concatenates output features; a row split sums partial outputs. Both match a serial matrix-vector product.
4. **Context partitioning:** query and key/value blocks cover the sequence once; stable local softmax statistics merge to the exact causal-attention forward result.

An optional [PyTorch two-rank lab](src/distributed_lab/torch_distributed.py) adds **real Gloo collectives on two CPU processes** for a small column-then-row tensor-parallel FFN. Each rank owns half the hidden features. The lab compares the collective output, both weight-gradient shards, summed input gradient, and one sharded SGD step with serial autograd. This is a correctness check, not a speed or memory result. Supplementary reference modules now demonstrate ZeRO-style optimizer ownership, FSDP-style gather/gradient ownership, 1F1B and split-backward dependency clocks, GQA/MLA head partitioning, expert tensor parallelism, mesh groups, and atomic local multi-shard checkpoint/resume. They execute in one CPU process and do not implement production ZeRO/FSDP, Zero Bubble, DualPipe, or an NCCL training mesh. A small optional CUDA profiler/AMP audit can collect hardware evidence when run on a CUDA GPU; the checked-in tests establish CPU behavior only. The standard-library pipeline schedule is a sequential **fill-drain event trace**, and the context exercise tests blockwise softmax algebra without Ring Attention's communication. See [LESSON_MAP.md](LESSON_MAP.md) for the lesson-by-lesson scope.

A second [PyTorch DDP lab](src/distributed_lab/ddp_decoder.py) trains a one-block causal decoder on two CPU/Gloo ranks. It compares the average of two equal-sized local next-token losses, every synchronized parameter gradient, and one SGD update against a serial full-batch model. A causal-prefix test confirms that later tokens cannot change earlier logits. This is a numerical correctness exercise, not a GPU throughput result.

Two more [process-parallel checks](src/distributed_lab/process_parallel.py) use actual point-to-point Gloo communication. The pipeline check splits a small network across two processes, sends two microbatch activations forward and their gradients backward, and compares output, parameter gradients, and an SGD step to serial autograd. The context check assigns each process two consecutive query/key/value positions, exchanges key/value blocks, returns remote key/value gradients to their owner, and compares causal-attention output and all input gradients to serial attention. Its online softmax merge is blockwise, but this two-rank synchronous exchange is **not** Ring Attention's asynchronous, computation-overlapped schedule.

The [expert-parallel check](src/distributed_lab/expert_parallel.py) assigns two linear experts to each of two ranks. Five tokens request their top two experts; each expert accepts at most two assignments in token order, so two requests are dropped. The source rank dispatches token vectors to the remote expert owner, combines weighted outputs, and sends output gradients back; the owner returns token-input gradients. Forward values, expert/router/input gradients, and one update match a serial reference. Gate weights use a softmax over each token's original top two scores and are **not renormalized after capacity drops**. This is a fixed routing exercise, not DeepSeek-V3's gate, load-balancing strategy, all-to-all implementation, or throughput profile.

## Run

Use Python 3.10 or newer from this directory; installation and a GPU are unnecessary:

```bash
PYTHONPATH=src python3 -m distributed_lab.checks
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The first command prints four booleans and exits nonzero if any parity check fails. The second runs the invariant suite. Start at `src/distributed_lab/partitions.py`, then read `data.py`, `pipeline.py`, `tensor.py`, and `context.py`. Change a partition or reduction rule deliberately and use the tests to identify the broken contract.

For the real two-process lab, install the optional dependency in a virtual environment and run:

```bash
python3 -m pip install -e '.[torch]'
python3 -m distributed_lab.torch_distributed
python3 -m distributed_lab.ddp_decoder
python3 -m distributed_lab.process_parallel
python3 -m distributed_lab.expert_parallel
python3 -m unittest discover -s tests -v
```

These labs use a fresh file rendezvous, Gloo, `float64` tensors, and two spawned CPU workers. On macOS they default to the `lo0` loopback interface unless `GLOO_SOCKET_IFNAME` is already set. They need permission to bind a local socket. Failures in either rank make the parent command fail. The standard-library suite still runs when PyTorch is absent; optional integration tests skip in that case. The DDP comparison assumes two equal-sized shards and a mean loss on each rank; uneven shards require weighting by sample/token count rather than naively averaging local means.

## Supplementary contracts

`state.py` checks a parameter/optimizer ownership ledger and sharded Adam update parity. `schedules.py` validates operation dependencies, stage resource use, and saved-activation lifetimes; its F/B/W policy is a reference list scheduler, not a reproduction of published Zero Bubble or DualPipe schedules. `parallel_contracts.py` compares head/hidden/context partitions and their autograd gradients with a serial function, then enumerates independent mesh groups. Its joint example composes dense DP/TP/CP ownership and a two-stage function in one process; it does not compose EP or physical pipeline communication. `checkpoint.py` publishes checksummed local shard files by same-filesystem directory rename and rejects incomplete/corrupt versions. The tests compare exact six-step continuation with save at step three and resume, including Adam moments, RNG and data cursor. This local format is not an object-store or multi-host checkpoint service.

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
