# Distributed training contract lab

This is a **CPU-only, standard-library** student project for lectures 74–89 of *From random weights to your own LLM*. It makes four mathematical and ownership contracts executable before students introduce PyTorch distributed processes and GPUs:

1. **Data partitioning:** every sample has one owner; uneven local gradient sums must be reduced using the *global sample count*.
2. **Pipeline partitioning:** contiguous layer stages preserve forward and backward dependencies, and accumulated microbatch gradients equal a serial chain's gradients.
3. **Tensor partitioning:** a column split concatenates output features; a row split sums partial outputs. Both match a serial matrix-vector product.
4. **Context partitioning:** query and key/value blocks cover the sequence once; stable local softmax statistics merge to the exact causal-attention forward result.

No network collectives, accelerator kernels, optimizer, activation checkpointing, ZeRO/FSDP, 1F1B, Zero Bubble, DualPipe, expert parallelism, distributed checkpoint, or performance measurement are implemented here. The pipeline schedule is a sequential **fill-drain event trace**, and the context exercise tests blockwise softmax algebra without Ring Attention's communication. See [LESSON_MAP.md](LESSON_MAP.md) for the lesson-by-lesson scope.

## Run

Use Python 3.10 or newer from this directory; installation and a GPU are unnecessary:

```bash
PYTHONPATH=src python3 -m distributed_lab.checks
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The first command prints four booleans and exits nonzero if any parity check fails. The second runs the invariant suite. Start at `src/distributed_lab/partitions.py`, then read `data.py`, `pipeline.py`, `tensor.py`, and `context.py`. Change a partition or reduction rule deliberately and use the tests to identify the broken contract.

## Next implementation steps

Replace each CPU simulation with a separate PyTorch distributed exercise, preserving its serial parity test: initialize process groups, send tensors between stages, implement real collectives, compare forward **and backward** numerics, and only then measure memory/throughput. Use tiny inputs for correctness first. Real DDP averages gradients according to its documented semantics; the toy here uses summed local sample gradients divided by the global count to stay correct for uneven shard sizes. A production implementation needs to handle sampler behavior, loss reduction, and accumulation consistently.

## Primary references

- [PyTorch DistributedDataParallel documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [distributed communication primitives](https://docs.pytorch.org/docs/stable/distributed/) for actual process groups and reductions.
- [GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965) for microbatch pipeline training.
- [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) for tensor, pipeline, and data parallel composition.
- [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) for blockwise context-parallel attention.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) and [PyTorch FSDP documentation](https://docs.pytorch.org/docs/stable/fsdp.html) for sharded training state.
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) and the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) for advanced pipeline scheduling and DualPipe.

These references motivate future exercises. Passing this lab does **not** validate any of their full algorithms or performance claims.
