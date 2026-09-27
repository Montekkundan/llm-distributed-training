"""CPU matrix-vector contracts for column and row tensor partitioning."""

from __future__ import annotations

from dataclasses import dataclass

from .partitions import balanced_ranges


@dataclass(frozen=True)
class TensorParallelResult:
    output: tuple[float, ...]
    shards: tuple[tuple[int, ...], ...]
    partial_outputs: tuple[tuple[float, ...], ...]


def _shape(matrix: list[list[float]], vector: list[float]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix needs at least one row and column")
    columns = len(matrix[0])
    if len(vector) != columns or any(len(row) != columns for row in matrix):
        raise ValueError("Matrix must be rectangular and match vector length")
    return len(matrix), columns


def serial_matvec(matrix: list[list[float]], vector: list[float]) -> tuple[float, ...]:
    _shape(matrix, vector)
    return tuple(sum(value * vector[column] for column, value in enumerate(row)) for row in matrix)


def column_parallel_matvec(
    matrix: list[list[float]], vector: list[float], ranks: int,
) -> TensorParallelResult:
    """Split output rows, then concatenate local outputs in rank order."""
    rows, _ = _shape(matrix, vector)
    shards = balanced_ranges(rows, ranks, require_nonempty=True)
    partials = tuple(
        tuple(sum(matrix[row][column] * vector[column] for column in range(len(vector))) for row in shard)
        for shard in shards
    )
    return TensorParallelResult(
        output=tuple(value for partial in partials for value in partial),
        shards=tuple(tuple(shard) for shard in shards),
        partial_outputs=partials,
    )


def row_parallel_matvec(
    matrix: list[list[float]], vector: list[float], ranks: int,
) -> TensorParallelResult:
    """Split input columns, then sum rank-local partial outputs."""
    rows, columns = _shape(matrix, vector)
    shards = balanced_ranges(columns, ranks, require_nonempty=True)
    partials = tuple(
        tuple(sum(matrix[row][column] * vector[column] for column in shard) for row in range(rows))
        for shard in shards
    )
    return TensorParallelResult(
        output=tuple(sum(partial[row] for partial in partials) for row in range(rows)),
        shards=tuple(tuple(shard) for shard in shards),
        partial_outputs=partials,
    )
