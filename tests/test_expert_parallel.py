"""Real two-rank expert dispatch and combine parity."""

from __future__ import annotations

import unittest


class ExpertRoutingTests(unittest.TestCase):
    def test_capacity_and_top_k_contract(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch dependency is unavailable")
        from distributed_lab.expert_parallel import route

        logits = torch.tensor([[2., 1., 0., -1.],
                               [2., 1., 0., -1.],
                               [2., 0., 1., -1.],
                               [2., -1., 0., 1.],
                               [-1., 0., 2., 1.]])
        accepted, probabilities, counts = route(logits, top_k=2, capacity=2)
        self.assertEqual(counts, (2, 2, 2, 2))
        self.assertEqual(len(accepted), 8)
        self.assertIn((2, 1, 2), accepted)
        self.assertNotIn((2, 0, 0), accepted)
        self.assertIn((3, 1, 3), accepted)
        self.assertNotIn((3, 0, 0), accepted)
        self.assertLess(float(probabilities[2, 1]), 1.0)

    def test_rejects_invalid_route_contract(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch dependency is unavailable")
        from distributed_lab.expert_parallel import route

        logits = torch.zeros((2, 4))
        for top_k, capacity in ((0, 2), (5, 2), (2, 0), (True, 2)):
            with self.subTest(top_k=top_k, capacity=capacity), self.assertRaises(ValueError):
                route(logits, top_k=top_k, capacity=capacity)

    def test_two_rank_forward_backward_and_step_match_serial(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            self.skipTest("optional PyTorch dependency is unavailable")
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("PyTorch Gloo backend is unavailable")
        from distributed_lab.expert_parallel import run_expert_parity

        result = run_expert_parity()
        self.assertEqual((result["backend"], result["world_size"]), ("gloo", 2))
        self.assertEqual(result["expert_counts"], [2, 2, 2, 2])
        self.assertEqual(result["dropped_assignments"], 2)
        for key in ("forward_error", "gradient_error", "post_step_error"):
            self.assertLess(result[key], 1e-10, key)


if __name__ == "__main__":
    unittest.main()
