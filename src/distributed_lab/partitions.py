"""Deterministic ownership plans shared by the CPU contract simulators."""

from __future__ import annotations


def balanced_ranges(length: int, ranks: int, *, require_nonempty: bool = False) -> tuple[range, ...]:
    """Contiguous, disjoint ranges whose concatenation is range(length)."""
    if type(length) is not int or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if type(ranks) is not int or ranks <= 0:
        raise ValueError("ranks must be a positive integer")
    if require_nonempty and length < ranks:
        raise ValueError("Each rank needs at least one item")
    width, remainder = divmod(length, ranks)
    sizes = [width + (rank < remainder) for rank in range(ranks)]
    starts = [0]
    for size in sizes:
        starts.append(starts[-1] + size)
    return tuple(range(starts[rank], starts[rank + 1]) for rank in range(ranks))
