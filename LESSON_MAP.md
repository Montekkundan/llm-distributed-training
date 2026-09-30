# Lectures 74–89: code and exercise map

The four standard-library exercises are narrow CPU contracts; optional PyTorch labs run two-rank DDP, tensor-parallel FFN, two-stage pipeline, context-block exchange, and expert dispatch/combine checks. “Future” below means **not implemented** in this starter. Supplementary CPU references are single-process correctness demonstrations, distinct from the existing real two-rank Gloo checks.

| Lecture | Topic | File / command | Scope |
| --- | --- | --- | --- |
| 74 | GPU profiling | `src/distributed_lab/profiling.py` | Implemented: repeatable CPU/CUDA baseline, synchronized timing, memory metadata and optional trace. CUDA throughput unmeasured in course validation. |
| 75 | Mixed precision and recomputation | `src/distributed_lab/profiling.py` | Implemented: AMP attention/reduction audit and checkpoint-sequential benchmark. CPU smoke checked; GPU precision/memory tradeoffs require CUDA results. |
| 76 | DDP | `src/distributed_lab/data.py`; `python3 -m distributed_lab.ddp_decoder` with optional PyTorch installed | Implemented: uneven-sample algebra and real two-rank CPU/Gloo causal-decoder gradient/update parity for equal shards. Future: distributed sampler, uneven shards, GPU/NCCL. |
| 77 | ZeRO | `src/distributed_lab/state.py` | Implemented: byte ledger and sharded Adam reference/update parity. Future: production ZeRO communication. |
| 78 | FSDP | `src/distributed_lab/state.py` | Implemented: parameter gather and owned-gradient reference parity. Future: real FSDP training and peak-memory measurement. |
| 79 | Tensor-parallel FFN | `src/distributed_lab/tensor.py`; `python3 -m distributed_lab.torch_distributed` with optional PyTorch installed | Implemented: row/column matrix-vector algebra plus a two-rank CPU/Gloo sharded FFN with real output/input-gradient all-reduces and serial forward/backward/update parity. Future: GPU/NCCL profiling and broader shapes. |
| 80 | Tensor-parallel attention and MLA | `src/distributed_lab/parallel_contracts.py` | Implemented: GQA whole-head layout and MLA non-positional latent projection forward/backward parity. Future: physical groups, decoupled RoPE and cache absorption in TP. |
| 81 | GPipe | `src/distributed_lab/pipeline.py`; `python3 -m distributed_lab.process_parallel` | Implemented: contiguous stages, fill-drain event order, two-process activation/gradient sends, serial gradient/update parity. Future: overlapped schedule and activation recomputation. |
| 82 | 1F1B | `src/distributed_lab/schedules.py` | Implemented: dependency-valid 1F1B clock and activation-liveness accounting. Future: measured multi-GPU 1F1B. |
| 83 | Zero Bubble and DualPipe | `src/distributed_lab/schedules.py` | Implemented: legal delayed-weight-backward reference clock. Future: source-faithful Zero Bubble/DualPipe and actual overlap. |
| 84 | Context parallelism and Ring Attention | `src/distributed_lab/context.py`; `python3 -m distributed_lab.process_parallel` | Implemented: two-process key/value exchange and remote gradient return with causal forward/backward parity. Future: asynchronous ring overlap, GPUs, long sequences. |
| 85 | Long-context training | `src/distributed_lab/context.py`; `src/distributed_lab/parallel_contracts.py` | Implemented: global-position causal query partition parity. Future: document-packed long-context GPU training/memory/throughput. |
| 86 | Expert parallelism | `python3 -m distributed_lab.expert_parallel` | Implemented: two-process expert ownership, top-2/capacity routing, dispatch/combine, serial forward/gradient/update parity. Future: learned router, load balancing, all-to-all and GPU throughput. |
| 87 | Expert tensor parallelism | `src/distributed_lab/parallel_contracts.py` | Implemented: unequal SwiGLU hidden-slice forward/backward parity. Future: physical nested EP/ETP groups. |
| 88 | Compose DP/TP/PP/CP/EP | `src/distributed_lab/parallel_contracts.py` | Implemented: mesh group enumeration and joint dense DP/TP/CP/stage reference forward/backward parity. Future: one physical DP/TP/PP/CP/EP topology. |
| 89 | Distributed checkpoint/resume | `src/distributed_lab/checkpoint.py` | Implemented: local checksummed multi-shard atomic commit, incomplete rejection and exact resume parity. Future: real multi-host storage/resharding/failure recovery. |

The standard-library modules have tests in `tests/test_contracts.py`; the optional two-process labs have `tests/test_torch_distributed.py`, `tests/test_ddp_decoder.py`, `tests/test_stage_context.py`, and `tests/test_expert_parallel.py`. Supplementary contracts, negative cases and checkpoint/resume live in `tests/test_references.py`. The table is a teaching map, not a claim that the complete distributed training framework is present.
