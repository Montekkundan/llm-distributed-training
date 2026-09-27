"""Uneven data shards and sample-count-correct gradient reduction."""

from __future__ import annotations

from dataclasses import dataclass

from .partitions import balanced_ranges


@dataclass(frozen=True)
class DataParallelResult:
    shard_indices: tuple[tuple[int, ...], ...]
    local_gradient_sums: tuple[float, ...]
    global_mean_gradient: float
    global_mean_loss: float


def serial_linear_gradient(samples: list[tuple[float, float]], weight: float) -> tuple[float, float]:
    """Mean derivative and mean half-squared loss for y_hat = weight * x."""
    if not samples:
        raise ValueError("At least one sample is required")
    gradient = 0.0
    loss = 0.0
    for x, target in samples:
        error = weight * x - target
        gradient += error * x
        loss += 0.5 * error * error
    return gradient / len(samples), loss / len(samples)


def data_parallel_linear_gradient(
    samples: list[tuple[float, float]], weight: float, ranks: int,
) -> DataParallelResult:
    """Simulate shard ownership and a weighted gradient all-reduce, not DDP itself."""
    if not samples:
        raise ValueError("At least one sample is required")
    shards = balanced_ranges(len(samples), ranks)
    local_sums: list[float] = []
    loss_sum = 0.0
    for shard in shards:
        gradient_sum = 0.0
        for index in shard:
            x, target = samples[index]
            error = weight * x - target
            gradient_sum += error * x
            loss_sum += 0.5 * error * error
        local_sums.append(gradient_sum)
    return DataParallelResult(
        shard_indices=tuple(tuple(shard) for shard in shards),
        local_gradient_sums=tuple(local_sums),
        global_mean_gradient=sum(local_sums) / len(samples),
        global_mean_loss=loss_sum / len(samples),
    )
