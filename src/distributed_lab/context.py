"""Blockwise causal-attention forward contract; no ring transport is modeled."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

from .partitions import balanced_ranges


@dataclass(frozen=True)
class ContextParallelResult:
    output: tuple[float, ...]
    query_shards: tuple[tuple[int, ...], ...]
    key_value_shards: tuple[tuple[int, ...], ...]


def _validate(queries: list[float], keys: list[float], values: list[float]) -> None:
    if not queries or len(queries) != len(keys) or len(keys) != len(values):
        raise ValueError("Queries, keys, and values need the same nonzero length")


def serial_causal_attention(
    queries: list[float], keys: list[float], values: list[float],
) -> tuple[float, ...]:
    """Exact stable softmax reference for one scalar head."""
    _validate(queries, keys, values)
    output = []
    for query_index, query in enumerate(queries):
        scores = [query * keys[key_index] for key_index in range(query_index + 1)]
        score_max = max(scores)
        weights = [exp(score - score_max) for score in scores]
        output.append(
            sum(weight * values[index] for index, weight in enumerate(weights)) / sum(weights)
        )
    return tuple(output)


def context_parallel_causal_attention(
    queries: list[float], keys: list[float], values: list[float], ranks: int,
) -> ContextParallelResult:
    """Merge key-block softmax statistics for query blocks, including the causal mask.

    This tests the blockwise algebra only. It does not implement Ring Attention's
    communication, backward pass, or sequence-parallel GPU kernels.
    """
    _validate(queries, keys, values)
    query_shards = balanced_ranges(len(queries), ranks, require_nonempty=True)
    key_shards = balanced_ranges(len(keys), ranks, require_nonempty=True)
    output = [0.0] * len(queries)
    for query_shard in query_shards:
        for query_index in query_shard:
            global_max = float("-inf")
            global_weight = 0.0
            global_weighted_value = 0.0
            for key_shard in key_shards:
                allowed = [key_index for key_index in key_shard if key_index <= query_index]
                if not allowed:
                    continue
                scores = [queries[query_index] * keys[key_index] for key_index in allowed]
                local_max = max(scores)
                local_weights = [exp(score - local_max) for score in scores]
                local_weight = sum(local_weights)
                local_weighted_value = sum(
                    weight * values[key_index]
                    for key_index, weight in zip(allowed, local_weights)
                )
                merged_max = max(global_max, local_max)
                old_scale = exp(global_max - merged_max) if global_weight else 0.0
                new_scale = exp(local_max - merged_max)
                global_weight = global_weight * old_scale + local_weight * new_scale
                global_weighted_value = (
                    global_weighted_value * old_scale + local_weighted_value * new_scale
                )
                global_max = merged_max
            output[query_index] = global_weighted_value / global_weight
    return ContextParallelResult(
        output=tuple(output),
        query_shards=tuple(tuple(shard) for shard in query_shards),
        key_value_shards=tuple(tuple(shard) for shard in key_shards),
    )
