# Lectures 74–89: code and exercise map

The four standard-library exercises are narrow CPU contracts; optional PyTorch labs run two-rank DDP, tensor-parallel FFN, two-stage pipeline, context-block exchange, and expert dispatch/combine checks. “Future” below means **not implemented** in this starter.

| Lecture | Topic | File / command | Scope |
| --- | --- | --- | --- |
| 74 | GPU profiling | `README.md` | Future: use actual profiler traces and measured bottlenecks. |
| 75 | Mixed precision and recomputation | `README.md` | Future: numeric tolerance, memory accounting, activation checkpointing. |
| 76 | DDP | `src/distributed_lab/data.py`; `python3 -m distributed_lab.ddp_decoder` with optional PyTorch installed | Implemented: uneven-sample algebra and real two-rank CPU/Gloo causal-decoder gradient/update parity for equal shards. Future: distributed sampler, uneven shards, GPU/NCCL. |
| 77 | ZeRO | `README.md` | Future: shard optimizer state and verify parity after an update. |
| 78 | FSDP | `README.md` | Future: shard parameters/gradients and model state. |
| 79 | Tensor-parallel FFN | `src/distributed_lab/tensor.py`; `python3 -m distributed_lab.torch_distributed` with optional PyTorch installed | Implemented: row/column matrix-vector algebra plus a two-rank CPU/Gloo sharded FFN with real output/input-gradient all-reduces and serial forward/backward/update parity. Future: GPU/NCCL profiling and broader shapes. |
| 80 | Tensor-parallel attention and MLA | `src/distributed_lab/tensor.py` | Foundation only: linear partition contracts. Future: attention/MLA projections and backward parity. |
| 81 | GPipe | `src/distributed_lab/pipeline.py`; `python3 -m distributed_lab.process_parallel` | Implemented: contiguous stages, fill-drain event order, two-process activation/gradient sends, serial gradient/update parity. Future: overlapped schedule and activation recomputation. |
| 82 | 1F1B | `README.md` | Future: dependency-valid interleaved schedule and bubble accounting. |
| 83 | Zero Bubble and DualPipe | `README.md` | Future: split backward work and bidirectional schedule; compare against primary papers. |
| 84 | Context parallelism and Ring Attention | `src/distributed_lab/context.py`; `python3 -m distributed_lab.process_parallel` | Implemented: two-process key/value exchange and remote gradient return with causal forward/backward parity. Future: asynchronous ring overlap, GPUs, long sequences. |
| 85 | Long-context training | `src/distributed_lab/context.py` | Foundation only: causal block correctness. Future: position handling, memory, throughput, training. |
| 86 | Expert parallelism | `python3 -m distributed_lab.expert_parallel` | Implemented: two-process expert ownership, top-2/capacity routing, dispatch/combine, serial forward/gradient/update parity. Future: learned router, load balancing, all-to-all and GPU throughput. |
| 87 | Expert tensor parallelism | `README.md` | Future: combine expert and tensor partitions with numerical parity. |
| 88 | Compose DP/TP/PP/CP/EP | All four modules | Separate contracts only; future: one jointly executable topology and global reference parity. |
| 89 | Distributed checkpoint/resume | `README.md` | Future: shard manifest, restore, and bitwise/within-tolerance continuation test. |

The standard-library modules have tests in `tests/test_contracts.py`; the optional two-process labs have `tests/test_torch_distributed.py`, `tests/test_ddp_decoder.py`, `tests/test_stage_context.py`, and `tests/test_expert_parallel.py`. The table is a teaching map, not a claim that the complete distributed training framework is present.
