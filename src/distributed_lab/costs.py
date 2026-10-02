"""Closed-form cost models for the distributed-training lessons (pure Python).

Each function states its equation and the assumptions under which it holds.
Nothing here is measured. The formulas are idealised models: no latency
hiding, no contention, no kernel launch cost. ``tests/test_costs.py`` checks
them against hand-computed values and against the CPU reference simulators in
``schedules.py`` and ``state.py``.

Run ``PYTHONPATH=src python3 -m distributed_lab.costs`` to print worked examples.
The hardware numbers there are round illustrative values, not any named device.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative(value, name):
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


# --- Collective communication (lessons 76-79, 86) ---------------------------
#
# Ring model. N ranks sit on a ring and every rank sends to its successor on
# each step, so all links are busy at once. S is the message size in bytes on
# every rank. Links are uniform and full duplex, and reduction arithmetic is
# free. A step costs alpha + (S / N) * beta, where alpha is seconds per message
# (latency) and beta is seconds per byte (1 / bandwidth).


def ring_reduce_scatter_bytes(size_bytes, ranks):
    """Bytes each rank sends: (N - 1) / N * S, over N - 1 steps of S / N bytes."""
    _positive_int(ranks, "ranks")
    _nonnegative(size_bytes, "size_bytes")
    return (ranks - 1) / ranks * size_bytes


def ring_all_gather_bytes(size_bytes, ranks):
    """Bytes each rank sends: (N - 1) / N * S, where S is the gathered result."""
    return ring_reduce_scatter_bytes(size_bytes, ranks)


def ring_all_reduce_bytes(size_bytes, ranks):
    """Bytes each rank sends: 2 (N - 1) / N * S (reduce-scatter, then all-gather).

    The value is below 2 S for every N and does not grow with N, which is why
    ring all-reduce is bandwidth-optimal. N = 2 gives exactly S.
    """
    return (ring_reduce_scatter_bytes(size_bytes, ranks)
            + ring_all_gather_bytes(size_bytes, ranks))


def ring_reduce_scatter_time(size_bytes, ranks, *, alpha, beta):
    """(N - 1) alpha + (N - 1) / N * S * beta seconds."""
    volume = ring_reduce_scatter_bytes(size_bytes, ranks)
    _nonnegative(alpha, "alpha")
    _nonnegative(beta, "beta")
    return (ranks - 1) * alpha + volume * beta


def ring_all_gather_time(size_bytes, ranks, *, alpha, beta):
    """(N - 1) alpha + (N - 1) / N * S * beta seconds."""
    return ring_reduce_scatter_time(size_bytes, ranks, alpha=alpha, beta=beta)


def ring_all_reduce_time(size_bytes, ranks, *, alpha, beta):
    """2 (N - 1) alpha + 2 (N - 1) / N * S * beta seconds.

    Latency grows with N while the bandwidth term saturates, so small messages
    on many ranks are latency bound.
    """
    return (ring_reduce_scatter_time(size_bytes, ranks, alpha=alpha, beta=beta)
            + ring_all_gather_time(size_bytes, ranks, alpha=alpha, beta=beta))


def all_to_all_bytes(size_bytes, ranks):
    """Bytes each rank sends: (N - 1) / N * S.

    Assumes equal splits, so each of the N - 1 peers receives S / N bytes. With
    unequal splits (expert routing) the busiest rank sets the time.
    """
    return ring_reduce_scatter_bytes(size_bytes, ranks)


# --- Tensor parallelism (lessons 79-80) -------------------------------------


@dataclass(frozen=True)
class TensorParallelBlockCost:
    forward_all_reduces: int
    backward_all_reduces: int
    message_bytes: float
    ring_bytes_per_rank: float


def tensor_parallel_block_cost(batch, tokens, hidden, tp, *, bytes_per_element=2):
    """Megatron-style collectives for one transformer block.

    Attention (column-parallel QKV, row-parallel output) and the MLP
    (column-parallel up, row-parallel down) each need one all-reduce of the
    activation tensor [batch, tokens, hidden] in the forward pass (summing row
    partial outputs) and one in the backward pass (summing the input gradient).
    That is 2 forward and 2 backward all-reduces per block. With a ring each
    moves 2 (t - 1) / t * M bytes per rank, M = batch * tokens * hidden * bytes,
    so the block total is 8 (t - 1) / t * M.

    Assumptions: no sequence parallelism (that replaces each all-reduce by a
    reduce-scatter plus an all-gather of the same total volume), and no
    activation recomputation (recomputing a block repeats its 2 forward
    all-reduces during the backward pass).
    """
    for name, value in (("batch", batch), ("tokens", tokens), ("hidden", hidden),
                        ("tp", tp), ("bytes_per_element", bytes_per_element)):
        _positive_int(value, name)
    message = batch * tokens * hidden * bytes_per_element
    return TensorParallelBlockCost(2, 2, message, 4 * ring_all_reduce_bytes(message, tp))


# --- Pipeline parallelism (lessons 81-83) -----------------------------------
#
# p stages (one per device), m microbatches, equal per-stage forward time F and
# backward time B, no communication cost.


def pipeline_makespan(stages, microbatches, forward, backward):
    """(m + p - 1) (F + B) for GPipe and for non-interleaved 1F1B.

    Both fill and drain the pipe once; they differ in saved activations, not in
    time (see ``gpipe_peak_microbatches`` and ``one_f_one_b_peak_microbatches``).
    """
    _positive_int(stages, "stages")
    _positive_int(microbatches, "microbatches")
    return (microbatches + stages - 1) * (forward + backward)


def pipeline_bubble_fraction(stages, microbatches):
    """(p - 1) / (m + p - 1), the idle share of every stage.

    It follows from 1 - m (F + B) / ((m + p - 1) (F + B)) and does not depend
    on F or B. It shrinks as m grows relative to p.
    """
    _positive_int(stages, "stages")
    _positive_int(microbatches, "microbatches")
    return (stages - 1) / (microbatches + stages - 1)


def gpipe_peak_microbatches(microbatches):
    """GPipe finishes all forwards before any backward, so every stage holds m."""
    _positive_int(microbatches, "microbatches")
    return microbatches


def one_f_one_b_peak_microbatches(stage, stages, microbatches):
    """min(p - s, m) saved inputs on stage s (0 is first), the 1F1B warm-up depth."""
    _positive_int(stages, "stages")
    _positive_int(microbatches, "microbatches")
    if type(stage) is not int or not 0 <= stage < stages:
        raise ValueError("stage must be in range(stages)")
    return min(stages - stage, microbatches)


def split_backward_makespan_lower_bound(stages, microbatches, forward, backward_input, backward_weight):
    """(p - 1) F + m (F + B + W): a bound no schedule can beat.

    The last stage cannot start before the first microbatch has crossed the
    p - 1 earlier forwards, and it then has m (F + B + W) of work to do. A
    schedule that reaches the bound has no bubble beyond that unavoidable
    start-up delay. Splitting the backward into B (input gradient) and W
    (weight gradient) lets W fill the idle slots that 1F1B leaves.
    """
    _positive_int(stages, "stages")
    _positive_int(microbatches, "microbatches")
    return ((stages - 1) * forward
            + microbatches * (forward + backward_input + backward_weight))


# --- Optimizer-state memory (lessons 77-78) ---------------------------------


def zero_stage_bytes_per_rank(parameters, ranks, stage, *, param_bytes=2, grad_bytes=2,
                              optimizer_bytes=12):
    """Per-rank bytes of model state for ZeRO stage 0, 1, 2 or 3.

    With Psi parameters, N ranks and per-parameter bytes P (weights), G
    (gradients) and K (optimizer state):

        stage 0 (replicated, plain DDP):  (P + G + K) Psi
        stage 1 (shard optimizer state):  (P + G + K / N) Psi
        stage 2 (also shard gradients):   (P + (G + K) / N) Psi
        stage 3 (also shard weights):     (P + G + K) Psi / N

    The defaults are the ZeRO paper's mixed-precision Adam accounting: 16-bit
    weights and gradients (2 + 2) and K = 12 for the FP32 master copy plus Adam
    m and v (4 + 4 + 4), 16 bytes per parameter in total. ``state.zero_ledger``
    uses grad_bytes=4 (an FP32 gradient buffer), 18 bytes per parameter; call
    this with grad_bytes=4 to reproduce it. Activations, communication buffers,
    fragmentation and the transient gathered layer in stage 3 are excluded.
    """
    _positive_int(parameters, "parameters")
    _positive_int(ranks, "ranks")
    if stage not in (0, 1, 2, 3):
        raise ValueError("stage must be 0, 1, 2 or 3")
    shard_optimizer, shard_gradients, shard_weights = stage >= 1, stage >= 2, stage >= 3
    per_parameter = (param_bytes / (ranks if shard_weights else 1)
                     + grad_bytes / (ranks if shard_gradients else 1)
                     + optimizer_bytes / (ranks if shard_optimizer else 1))
    return per_parameter * parameters


# --- Activation recomputation (lesson 75) -----------------------------------


def checkpoint_sequential_recompute_fraction(layers, segments):
    """Share of layers re-run in the backward pass by torch's checkpoint_sequential.

    PyTorch splits the layers into ``segments`` pieces of ``layers // segments``,
    checkpoints all but the last piece (the last runs normally and keeps its
    activations), and re-runs each checkpointed piece once during the backward
    pass. So the fraction is segment_size * (segments - 1) / layers, which is 0
    for one segment and less than 1 whenever the layers split evenly.
    """
    _positive_int(layers, "layers")
    _positive_int(segments, "segments")
    if segments > layers:
        raise ValueError("segments must not exceed layers")
    return (layers // segments) * (segments - 1) / layers


def step_flops_with_recompute(forward_flops, *, recomputed_fraction=1.0, backward_to_forward=2.0):
    """F (1 + b + r): forward, backward at b times the forward, plus r re-run forwards.

    b = 2 holds for matmul layers, where the input gradient and the weight
    gradient each cost one forward. Elementwise and attention-softmax work
    deviates from it.
    """
    _nonnegative(forward_flops, "forward_flops")
    if not 0 <= recomputed_fraction <= 1:
        raise ValueError("recomputed_fraction must be in [0, 1]")
    return forward_flops * (1 + backward_to_forward + recomputed_fraction)


def recompute_step_overhead(recomputed_fraction=1.0, *, backward_to_forward=2.0):
    """r / (1 + b): the extra share of step compute, 1/3 for full recompute at b = 2."""
    if not 0 <= recomputed_fraction <= 1:
        raise ValueError("recomputed_fraction must be in [0, 1]")
    return recomputed_fraction / (1 + backward_to_forward)


# --- Arithmetic intensity and the roofline (lessons 22, 74) -----------------


def arithmetic_intensity(flops, bytes_moved):
    """FLOP per byte moved between main memory and the compute units."""
    _nonnegative(flops, "flops")
    if bytes_moved <= 0:
        raise ValueError("bytes_moved must be positive")
    return flops / bytes_moved


def ridge_point(peak_flops, bandwidth):
    """peak_flops / bandwidth in FLOP per byte; below it a kernel is memory bound."""
    if peak_flops <= 0 or bandwidth <= 0:
        raise ValueError("peak_flops and bandwidth must be positive")
    return peak_flops / bandwidth


def attainable_flops(intensity, peak_flops, bandwidth):
    """min(peak, intensity * bandwidth): the roofline ceiling for one kernel."""
    _nonnegative(intensity, "intensity")
    ridge_point(peak_flops, bandwidth)
    return min(peak_flops, intensity * bandwidth)


def matmul_intensity(m, k, n, *, bytes_per_element=2):
    """2 m k n / (b (m k + k n + m n)): each operand is read or written once.

    This is the best case with perfect on-chip reuse. A matrix-vector product
    (m = 1) lands near 1 FLOP per byte at 16-bit precision, which is why decoding
    is memory bound, while a large square matmul reaches n / 3 FLOP per byte.
    """
    for name, value in (("m", m), ("k", k), ("n", n), ("bytes_per_element", bytes_per_element)):
        _positive_int(value, name)
    return arithmetic_intensity(2 * m * k * n,
                                bytes_per_element * (m * k + k * n + m * n))


def main():
    gib = 2 ** 30
    peak, bandwidth = 1e15, 2e12  # illustrative round numbers, not a named device
    report = {
        "ring_all_reduce_bytes_8_ranks_1_GiB": ring_all_reduce_bytes(gib, 8),
        "tensor_parallel_block_ring_bytes_tp8_b2_s1024_h4096_bf16":
            tensor_parallel_block_cost(2, 1024, 4096, 8).ring_bytes_per_rank,
        "pipeline_bubble_fraction_p4_m8": pipeline_bubble_fraction(4, 8),
        "zero_bytes_per_rank_psi7.5e9_n64_stage0123":
            [zero_stage_bytes_per_rank(7_500_000_000, 64, stage) for stage in range(4)],
        "recompute_step_overhead_full": recompute_step_overhead(1.0),
        "illustrative_ridge_point_flop_per_byte": ridge_point(peak, bandwidth),
        "matmul_intensity_4096_cube_bf16": matmul_intensity(4096, 4096, 4096),
        "matvec_intensity_4096_bf16": matmul_intensity(1, 4096, 4096),
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
