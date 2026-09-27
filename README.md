# Distributed training contract lab

This is a **CPU-only** student project for lectures 74–89 of *From random weights to your own LLM*. Its four original mathematical and ownership contract exercises use only the standard library:

1. **Data partitioning:** every sample has one owner; uneven local gradient sums must be reduced using the *global sample count*.
2. **Pipeline partitioning:** contiguous layer stages preserve forward and backward dependencies, and accumulated microbatch gradients equal a serial chain's gradients.
3. **Tensor partitioning:** a column split concatenates output features; a row split sums partial outputs. Both match a serial matrix-vector product.
4. **Context partitioning:** query and key/value blocks cover the sequence once; stable local softmax statistics merge to the exact causal-attention forward result.

An optional [PyTorch two-rank lab](src/distributed_lab/torch_distributed.py) adds **real Gloo collectives on two CPU processes** for a small column-then-row tensor-parallel FFN. Each rank owns half the hidden features. The lab compares the collective output, both weight-gradient shards, summed input gradient, and one sharded SGD step with serial autograd. This is a correctness check, not a speed or memory result. No accelerator kernels, activation checkpointing, ZeRO/FSDP, 1F1B, Zero Bubble, DualPipe, expert parallelism, distributed checkpoint, or performance measurement are implemented here. The pipeline schedule is a sequential **fill-drain event trace**, and the context exercise tests blockwise softmax algebra without Ring Attention's communication. See [LESSON_MAP.md](LESSON_MAP.md) for the lesson-by-lesson scope.

A second [PyTorch DDP lab](src/distributed_lab/ddp_decoder.py) trains a one-block causal decoder on two CPU/Gloo ranks. It compares the average of two equal-sized local next-token losses, every synchronized parameter gradient, and one SGD update against a serial full-batch model. A causal-prefix test confirms that later tokens cannot change earlier logits. This is a numerical correctness exercise, not a GPU throughput result.

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
python3 -m unittest discover -s tests -v
```

Both labs use a fresh file rendezvous, Gloo, `float64` tensors, and two spawned CPU workers. On macOS they default to the `lo0` loopback interface unless `GLOO_SOCKET_IFNAME` is already set. They need permission to bind a local socket. Failures in either rank make the parent command fail. The standard-library suite still runs when PyTorch is absent; optional integration tests skip in that case. The DDP comparison assumes two equal-sized shards and a mean loss on each rank; uneven shards require weighting by sample/token count rather than naively averaging local means.

## Next implementation steps

Extend the two-rank labs to multi-GPU/NCCL with explicit device mapping and parity checks before measuring peak memory, throughput, and communication overlap on named hardware. Then add real pipeline send/receive, FSDP, context-ring, and expert-routing exercises. The tensor-parallel FFN replicates the same batch on each rank and manually supplies the gradient of the reduced output to the local partial; it is not a general autograd-aware distributed operator. The DDP lab uses a fixed equal-shard batch rather than a distributed sampler, checkpointing, mixed precision, or gradient accumulation. A production implementation needs to handle those details and uneven token counts explicitly.

## Primary references

- [PyTorch DistributedDataParallel documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [distributed communication primitives](https://docs.pytorch.org/docs/stable/distributed/) for actual process groups and reductions.
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) for microbatch pipeline training.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) for tensor, pipeline, and data parallel composition.
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) for blockwise context-parallel attention.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) and [PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html) for sharded training state.
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) and the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) for advanced pipeline scheduling and DualPipe.

These references motivate future exercises. Passing this lab does **not** validate any of their full algorithms or performance claims.
