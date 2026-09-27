# Lectures 74–89: code and exercise map

The four implemented exercises are narrow CPU contracts. “Future” below means **not implemented** in this starter.

| Lecture | Topic | File / command | Scope |
| --- | --- | --- | --- |
| 74 | GPU profiling | `README.md` | Future: use actual profiler traces and measured bottlenecks. |
| 75 | Mixed precision and recomputation | `README.md` | Future: numeric tolerance, memory accounting, activation checkpointing. |
| 76 | DDP | `src/distributed_lab/data.py`; `PYTHONPATH=src python3 -m distributed_lab.checks` | Implemented: uneven sample ownership and global-count gradient reduction; future: processes and all-reduce. |
| 77 | ZeRO | `README.md` | Future: shard optimizer state and verify parity after an update. |
| 78 | FSDP | `README.md` | Future: shard parameters/gradients and model state. |
| 79 | Tensor-parallel FFN | `src/distributed_lab/tensor.py`; `PYTHONPATH=src python3 -m unittest discover -s tests -v` | Implemented: row/column matrix-vector split and gather/sum algebra; future: batched FFN and collectives. |
| 80 | Tensor-parallel attention and MLA | `src/distributed_lab/tensor.py` | Foundation only: linear partition contracts. Future: attention/MLA projections and backward parity. |
| 81 | GPipe | `src/distributed_lab/pipeline.py` | Implemented: contiguous stages, fill-drain event order, serial gradient parity; future: pipeline processes and sends. |
| 82 | 1F1B | `README.md` | Future: dependency-valid interleaved schedule and bubble accounting. |
| 83 | Zero Bubble and DualPipe | `README.md` | Future: split backward work and bidirectional schedule; compare against primary papers. |
| 84 | Context parallelism and Ring Attention | `src/distributed_lab/context.py` | Implemented: exact causal forward result via blockwise stable-softmax merge; future: ring communication and backward. |
| 85 | Long-context training | `src/distributed_lab/context.py` | Foundation only: causal block correctness. Future: position handling, memory, throughput, training. |
| 86 | Expert parallelism | `README.md` | Future: routing ownership, dispatch/combine, capacity and load balance. |
| 87 | Expert tensor parallelism | `README.md` | Future: combine expert and tensor partitions with numerical parity. |
| 88 | Compose DP/TP/PP/CP/EP | All four modules | Separate contracts only; future: one jointly executable topology and global reference parity. |
| 89 | Distributed checkpoint/resume | `README.md` | Future: shard manifest, restore, and bitwise/within-tolerance continuation test. |

Each implemented file has a matching test in `tests/test_contracts.py`. The table is a teaching map, not a claim that the complete distributed training framework is present.
