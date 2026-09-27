"""Regression tests for ownership, numerical parity, and event ordering."""

from __future__ import annotations

import math
import unittest

from distributed_lab.context import context_parallel_causal_attention, serial_causal_attention
from distributed_lab.data import data_parallel_linear_gradient, serial_linear_gradient
from distributed_lab.partitions import balanced_ranges
from distributed_lab.pipeline import serial_chain_loss_gradient, simulate_gpipe
from distributed_lab.tensor import column_parallel_matvec, row_parallel_matvec, serial_matvec


class PartitionTests(unittest.TestCase):
    def test_balanced_ranges_cover_once_even_when_ragged(self) -> None:
        shards = balanced_ranges(10, 3)
        self.assertEqual([list(shard) for shard in shards], [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual([index for shard in shards for index in shard], list(range(10)))

    def test_empty_shard_policy(self) -> None:
        self.assertEqual([list(shard) for shard in balanced_ranges(2, 3)], [[0], [1], []])
        with self.assertRaises(ValueError):
            balanced_ranges(2, 3, require_nonempty=True)
        with self.assertRaises(ValueError):
            balanced_ranges(3, 0)


class DataTests(unittest.TestCase):
    def test_uneven_sample_weighted_gradient_and_loss(self) -> None:
        samples = [(1.0, 2.0), (-2.0, 4.0), (3.0, -1.0), (4.0, 3.0), (0.5, -2.0)]
        expected_gradient, expected_loss = serial_linear_gradient(samples, 0.3)
        result = data_parallel_linear_gradient(samples, 0.3, 3)
        self.assertEqual(result.shard_indices, ((0, 1), (2, 3), (4,)))
        self.assertEqual(sorted(index for shard in result.shard_indices for index in shard), list(range(5)))
        self.assertAlmostEqual(result.global_mean_gradient, expected_gradient)
        self.assertAlmostEqual(result.global_mean_loss, expected_loss)
        self.assertNotAlmostEqual(sum(result.local_gradient_sums) / 3, expected_gradient)

    def test_more_ranks_than_samples(self) -> None:
        result = data_parallel_linear_gradient([(2.0, 1.0)], 0.5, 3)
        self.assertEqual(result.shard_indices, ((0,), (), ()))
        self.assertAlmostEqual(result.global_mean_gradient, 0.0)


class PipelineTests(unittest.TestCase):
    def test_forward_backward_and_loss_match_serial(self) -> None:
        weights = [0.5, -1.2, 0.7, 1.5, 0.9]
        inputs = [1.0, -2.0, 0.3, 4.0]
        targets = [0.0, 1.0, -0.5, 2.0]
        expected_loss, expected_gradients = serial_chain_loss_gradient(weights, inputs, targets)
        result = simulate_gpipe(weights, inputs, targets, 3)
        self.assertEqual(result.stage_layers, ((0, 1), (2, 3), (4,)))
        self.assertAlmostEqual(result.mean_loss, expected_loss)
        for actual, expected in zip(result.mean_gradients, expected_gradients):
            self.assertAlmostEqual(actual, expected)
        phases = [event.phase for event in result.events]
        self.assertEqual(phases, ["forward"] * 12 + ["backward"] * 12)
        self.assertEqual(
            [(event.microbatch, event.stage) for event in result.events[:3]],
            [(0, 0), (0, 1), (0, 2)],
        )
        self.assertEqual(
            [(event.microbatch, event.stage) for event in result.events[12:15]],
            [(3, 2), (3, 1), (3, 0)],
        )

    def test_rejects_impossible_stage_count(self) -> None:
        with self.assertRaises(ValueError):
            simulate_gpipe([1.0], [1.0], [0.0], 2)


class TensorTests(unittest.TestCase):
    def test_row_and_column_partition_match_serial_with_ragged_axes(self) -> None:
        matrix = [[1.0, -2.0, 0.5], [3.0, 0.0, -1.0], [-2.0, 2.5, 4.0], [0.0, 1.0, 9.0], [2.0, 3.0, 5.0]]
        vector = [0.5, -1.0, 2.0]
        expected = serial_matvec(matrix, vector)
        column = column_parallel_matvec(matrix, vector, 3)
        row = row_parallel_matvec(matrix, vector, 2)
        self.assertEqual(column.shards, ((0, 1), (2, 3), (4,)))
        self.assertEqual(row.shards, ((0, 1), (2,)))
        for actual in (column.output, row.output):
            for value, expected_value in zip(actual, expected):
                self.assertAlmostEqual(value, expected_value)

    def test_invalid_shapes_and_ranks(self) -> None:
        with self.assertRaises(ValueError):
            serial_matvec([[1.0], [1.0, 2.0]], [1.0])
        with self.assertRaises(ValueError):
            row_parallel_matvec([[1.0]], [1.0], 2)


class ContextTests(unittest.TestCase):
    def test_blockwise_causal_softmax_matches_reference(self) -> None:
        queries = [0.2, -0.7, 1.1, 0.6, -1.2, 0.0, 2.0]
        keys = [2.0, 0.1, -3.0, 4.0, -0.5, 0.8, 1.0]
        values = [4.0, 2.0, -1.0, 0.5, 3.0, 8.0, -2.0]
        expected = serial_causal_attention(queries, keys, values)
        for ranks in (1, 2, 3, 7):
            result = context_parallel_causal_attention(queries, keys, values, ranks)
            self.assertEqual([index for shard in result.query_shards for index in shard], list(range(7)))
            self.assertEqual([index for shard in result.key_value_shards for index in shard], list(range(7)))
            for actual, reference in zip(result.output, expected):
                self.assertTrue(math.isclose(actual, reference, rel_tol=1e-12, abs_tol=1e-12))

    def test_future_values_do_not_affect_past_outputs(self) -> None:
        queries = [0.1, 0.3, 0.5, 0.7]
        keys = [1.0, 2.0, 3.0, 4.0]
        first = context_parallel_causal_attention(queries, keys, [1.0, 2.0, 3.0, 4.0], 2)
        changed = context_parallel_causal_attention(queries, keys, [1.0, 2.0, 999.0, 999.0], 2)
        self.assertEqual(first.output[:2], changed.output[:2])

    def test_large_scores_remain_finite(self) -> None:
        result = context_parallel_causal_attention([1000.0] * 4, [1.0, -1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], 2)
        self.assertTrue(all(math.isfinite(value) for value in result.output))


if __name__ == "__main__":
    unittest.main()
