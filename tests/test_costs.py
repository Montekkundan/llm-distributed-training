"""Hand-computed checks of the closed-form cost models, cross-checked with the simulators."""

import unittest

from distributed_lab import costs
from distributed_lab.schedules import activation_peaks, one_f_one_b, split_backward
from distributed_lab.state import zero_ledger

try:
    import torch
except ImportError:
    torch = None


class CollectiveTests(unittest.TestCase):
    def test_ring_bytes_per_rank(self):
        # N = 4, S = 1000: reduce-scatter and all-gather send 3/4 S, all-reduce 3/2 S.
        self.assertEqual(costs.ring_reduce_scatter_bytes(1000, 4), 750)
        self.assertEqual(costs.ring_all_gather_bytes(1000, 4), 750)
        self.assertEqual(costs.ring_all_reduce_bytes(1000, 4), 1500)
        # N = 2 sends exactly S; one rank sends nothing; large N approaches 2 S from below.
        self.assertEqual(costs.ring_all_reduce_bytes(1000, 2), 1000)
        self.assertEqual(costs.ring_all_reduce_bytes(1000, 1), 0)
        self.assertAlmostEqual(costs.ring_all_reduce_bytes(1024, 1024), 2 * 1023)
        self.assertLess(costs.ring_all_reduce_bytes(1000, 10 ** 6), 2000)
        self.assertEqual(costs.all_to_all_bytes(1000, 4), 750)

    def test_alpha_beta_time(self):
        # alpha = 1e-5 s, beta = 1e-9 s/B (1 GB/s), S = 1e6 B, N = 4.
        alpha, beta, size = 1e-5, 1e-9, 1_000_000
        # reduce-scatter: 3 steps of latency + 750000 B; all-reduce is twice that.
        self.assertAlmostEqual(costs.ring_reduce_scatter_time(size, 4, alpha=alpha, beta=beta),
                               3 * 1e-5 + 750_000 * 1e-9, places=15)
        self.assertAlmostEqual(costs.ring_all_gather_time(size, 4, alpha=alpha, beta=beta), 7.8e-4, places=15)
        self.assertAlmostEqual(costs.ring_all_reduce_time(size, 4, alpha=alpha, beta=beta), 1.56e-3, places=15)
        # With alpha = 0 the time is bytes * beta.
        self.assertAlmostEqual(costs.ring_all_reduce_time(size, 4, alpha=0, beta=beta),
                               costs.ring_all_reduce_bytes(size, 4) * beta, places=15)

    def test_tensor_parallel_block(self):
        # b = 1, s = 2, h = 3, 4-byte elements: M = 24 B, tp = 2 -> each all-reduce moves 24 B.
        small = costs.tensor_parallel_block_cost(1, 2, 3, 2, bytes_per_element=4)
        self.assertEqual((small.forward_all_reduces, small.backward_all_reduces), (2, 2))
        self.assertEqual(small.message_bytes, 24)
        self.assertEqual(small.ring_bytes_per_rank, 96)
        # b = 2, s = 1024, h = 4096, BF16, tp = 8: M = 16 MiB; 8 (t - 1) / t * M = 7 * 16 MiB.
        large = costs.tensor_parallel_block_cost(2, 1024, 4096, 8)
        self.assertEqual(large.message_bytes, 16 * 2 ** 20)
        self.assertEqual(large.ring_bytes_per_rank, 7 * 16 * 2 ** 20)
        self.assertEqual(costs.tensor_parallel_block_cost(2, 1024, 4096, 1).ring_bytes_per_rank, 0)

    def test_invalid_arguments(self):
        for call in (lambda: costs.ring_all_reduce_bytes(10, 0),
                     lambda: costs.ring_all_reduce_bytes(-1, 2),
                     lambda: costs.ring_all_reduce_time(10, 2, alpha=-1, beta=0),
                     lambda: costs.tensor_parallel_block_cost(1, 1, 1, 0)):
            with self.assertRaises(ValueError):
                call()


class PipelineTests(unittest.TestCase):
    def test_bubble_fraction_values(self):
        self.assertAlmostEqual(costs.pipeline_bubble_fraction(4, 8), 3 / 11)
        self.assertAlmostEqual(costs.pipeline_bubble_fraction(4, 4), 3 / 7)
        self.assertAlmostEqual(costs.pipeline_bubble_fraction(2, 1), 1 / 2)
        self.assertEqual(costs.pipeline_bubble_fraction(1, 5), 0)
        self.assertEqual(costs.pipeline_makespan(4, 8, 1, 2), 33)

    def test_1f1b_simulator_reproduces_makespan_and_bubble(self):
        # The dependency clock must give (m + p - 1)(F + B) and so the closed-form bubble.
        for stages in range(1, 7):
            for microbatches in range(1, 13):
                for forward, backward in ((1, 1), (1, 2), (2, 3)):
                    events = one_f_one_b(stages, microbatches,
                                         forward_ticks=forward, backward_ticks=backward)
                    makespan = max(event.end for event in events)
                    label = (stages, microbatches, forward, backward)
                    self.assertEqual(makespan, costs.pipeline_makespan(
                        stages, microbatches, forward, backward), label)
                    busy = microbatches * (forward + backward)
                    self.assertAlmostEqual(1 - busy / makespan,
                                           costs.pipeline_bubble_fraction(stages, microbatches),
                                           msg=str(label))

    def test_1f1b_has_gpipe_bubble_but_bounded_activation_memory(self):
        stages, microbatches = 4, 8
        peaks = activation_peaks(one_f_one_b(stages, microbatches), stages)
        self.assertEqual(peaks, [4, 3, 2, 1])
        self.assertEqual(peaks, [costs.one_f_one_b_peak_microbatches(stage, stages, microbatches)
                                 for stage in range(stages)])
        self.assertEqual(costs.gpipe_peak_microbatches(microbatches), 8)
        # Fewer microbatches than stages: the warm-up depth is capped by m.
        self.assertEqual(activation_peaks(one_f_one_b(4, 2), 4), [2, 2, 2, 1])
        self.assertEqual([costs.one_f_one_b_peak_microbatches(s, 4, 2) for s in range(4)], [2, 2, 2, 1])

    def test_split_backward_reaches_the_lower_bound_with_gpipe_memory(self):
        # Unit F, B, W. For m >= p the greedy split-backward policy meets
        # (p - 1) + 3 m, the bound that no schedule beats ...
        for stages in range(1, 7):
            for microbatches in range(stages, stages + 8):
                events = split_backward(stages, microbatches)
                makespan = max(event.end for event in events)
                bound = costs.split_backward_makespan_lower_bound(stages, microbatches, 1, 1, 1)
                self.assertEqual(makespan, bound, (stages, microbatches))
                # ... and it admits every forward early, so each stage holds all m inputs
                # until its delayed W. That is GPipe-like memory, not the 1F1B profile.
                self.assertEqual(activation_peaks(events, stages, split=True),
                                 [microbatches] * stages)
        self.assertEqual(costs.split_backward_makespan_lower_bound(4, 8, 1, 1, 1), 27)

    def test_split_backward_beats_1f1b_time_at_equal_total_backward(self):
        # 1F1B with a combined backward of 2 ticks versus split B = W = 1 tick: 33 vs 27.
        combined = max(e.end for e in one_f_one_b(4, 8, backward_ticks=2))
        split = max(e.end for e in split_backward(4, 8))
        self.assertEqual((combined, split), (33, 27))


class ZeroMemoryTests(unittest.TestCase):
    def test_paper_example(self):
        # Psi = 7.5e9, N = 64, K = 12: 120, 31.40625, 16.640625, 1.875 (decimal GB).
        psi, ranks = 7_500_000_000, 64
        values = [costs.zero_stage_bytes_per_rank(psi, ranks, stage) for stage in range(4)]
        expected = [120e9, 4 * psi + 12 * psi / 64, 2 * psi + 14 * psi / 64, 16 * psi / 64]
        for actual, wanted in zip(values, expected):
            self.assertAlmostEqual(actual, wanted, delta=1e-6 * wanted)
        self.assertAlmostEqual(values[1], 31.40625e9, delta=1)
        self.assertAlmostEqual(values[2], 16.640625e9, delta=1)
        self.assertAlmostEqual(values[3], 1.875e9, delta=1)

    def test_state_ledger_is_the_same_model_with_fp32_gradients(self):
        # state.py: 2 (BF16 weights) + 4 (FP32 gradient) + 12 (FP32 master, m, v) = 18 B/param.
        self.assertEqual(zero_ledger(500_000_000, 4), (9e9, 4.5e9, 3e9, 2.25e9))
        for psi, ranks in ((500_000_000, 4), (123_456_789, 7), (1000, 1)):
            ledger = zero_ledger(psi, ranks)
            ours = [costs.zero_stage_bytes_per_rank(psi, ranks, stage, grad_bytes=4)
                    for stage in range(4)]
            paper = [costs.zero_stage_bytes_per_rank(psi, ranks, stage) for stage in range(4)]
            for index in range(4):
                self.assertAlmostEqual(ledger[index], ours[index], delta=1e-9 * ours[index])
            # The whole gap to the 16 B/param accounting is the 2 extra gradient bytes:
            # replicated in stages 0 and 1, sharded in stages 2 and 3.
            gaps = [ledger[index] - paper[index] for index in range(4)]
            for gap, share in zip(gaps, (1, 1, 1 / ranks, 1 / ranks)):
                self.assertAlmostEqual(gap, 2 * psi * share, delta=1e-9 * psi)

    def test_stage_ordering_and_invalid_input(self):
        values = [costs.zero_stage_bytes_per_rank(10 ** 9, 8, stage) for stage in range(4)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(costs.zero_stage_bytes_per_rank(10 ** 9, 1, 3),
                         costs.zero_stage_bytes_per_rank(10 ** 9, 1, 0))
        for stage in (-1, 4):
            with self.assertRaises(ValueError):
                costs.zero_stage_bytes_per_rank(10, 2, stage)


class RecomputeTests(unittest.TestCase):
    def test_overhead_values(self):
        # F = 1: no recompute costs 1 + 2 = 3, full recompute 1 + 2 + 1 = 4, so +1/3.
        self.assertEqual(costs.step_flops_with_recompute(1, recomputed_fraction=0), 3)
        self.assertEqual(costs.step_flops_with_recompute(1, recomputed_fraction=1), 4)
        self.assertAlmostEqual(costs.recompute_step_overhead(1.0), 1 / 3)
        self.assertAlmostEqual(costs.recompute_step_overhead(0.5), 1 / 6)
        self.assertAlmostEqual(costs.recompute_step_overhead(1.0, backward_to_forward=3.0), 1 / 4)
        self.assertEqual(costs.recompute_step_overhead(0.0), 0)
        with self.assertRaises(ValueError):
            costs.recompute_step_overhead(1.5)

    def test_checkpoint_sequential_fraction_values(self):
        self.assertEqual(costs.checkpoint_sequential_recompute_fraction(4, 1), 0)
        self.assertEqual(costs.checkpoint_sequential_recompute_fraction(4, 2), 0.5)
        self.assertEqual(costs.checkpoint_sequential_recompute_fraction(4, 4), 0.75)
        self.assertEqual(costs.checkpoint_sequential_recompute_fraction(12, 4), 0.75)
        self.assertAlmostEqual(costs.checkpoint_sequential_recompute_fraction(7, 3), 4 / 7)
        with self.assertRaises(ValueError):
            costs.checkpoint_sequential_recompute_fraction(2, 3)

    @unittest.skipIf(torch is None, "optional PyTorch is unavailable")
    def test_fraction_matches_layer_calls_in_torch(self):
        from torch import nn
        from torch.utils.checkpoint import checkpoint_sequential

        class Counted(nn.Module):
            calls = 0

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)

            def forward(self, x):
                Counted.calls += 1
                return self.linear(x)

        for layers, segments in ((4, 2), (4, 4), (7, 3), (12, 4)):
            model = nn.Sequential(*(Counted() for _ in range(layers)))
            Counted.calls = 0
            checkpoint_sequential(model, segments, torch.randn(2, 3, requires_grad=True),
                                  use_reentrant=False).sum().backward()
            fraction = costs.checkpoint_sequential_recompute_fraction(layers, segments)
            self.assertAlmostEqual(Counted.calls, layers * (1 + fraction), msg=str((layers, segments)))


class RooflineTests(unittest.TestCase):
    def test_ridge_and_attainable(self):
        # 1e15 FLOP/s over 2e12 B/s -> ridge at 500 FLOP/B.
        self.assertEqual(costs.ridge_point(1e15, 2e12), 500)
        self.assertEqual(costs.attainable_flops(100, 1e15, 2e12), 2e14)
        self.assertEqual(costs.attainable_flops(500, 1e15, 2e12), 1e15)
        self.assertEqual(costs.attainable_flops(5000, 1e15, 2e12), 1e15)
        self.assertEqual(costs.arithmetic_intensity(300, 100), 3)

    def test_matmul_intensity(self):
        # 4096^3 in BF16: 2 n^3 / (2 * 3 n^2) = n / 3.
        self.assertAlmostEqual(costs.matmul_intensity(4096, 4096, 4096), 4096 / 3)
        self.assertEqual(costs.matmul_intensity(4096, 4096, 4096) > costs.ridge_point(1e15, 2e12), True)
        # m = 1: 2 * 4096^2 FLOP over 2 * (4096 + 4096^2 + 4096) bytes, just under 1.
        vector = costs.matmul_intensity(1, 4096, 4096)
        self.assertAlmostEqual(vector, 2 * 4096 ** 2 / (2 * (4096 ** 2 + 2 * 4096)))
        self.assertLess(vector, 1)
        # 2x2x2 in 4-byte elements: 16 FLOP over 4 * 12 = 48 bytes.
        self.assertAlmostEqual(costs.matmul_intensity(2, 2, 2, bytes_per_element=4), 1 / 3)
        with self.assertRaises(ValueError):
            costs.ridge_point(0, 1)


if __name__ == "__main__":
    unittest.main()
