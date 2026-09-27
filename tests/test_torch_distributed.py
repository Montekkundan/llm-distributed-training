"""Integration checks for the optional real two-rank Gloo lab."""

from __future__ import annotations

import unittest

from distributed_lab.torch_distributed import run_two_rank_parity


class DistributedTorchTests(unittest.TestCase):
    def test_rejects_invalid_partition_without_spawning(self) -> None:
        for width in (0, 1, 3, True, 2.0):
            with self.subTest(width=width), self.assertRaises(ValueError):
                run_two_rank_parity(width)

    def test_two_real_gloo_ranks_match_serial_autograd_and_update(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            self.skipTest("optional PyTorch dependency is not installed")
        if not dist.is_available() or not dist.is_gloo_available():
            self.skipTest("PyTorch Gloo backend is not available")
        result = run_two_rank_parity(hidden_width=4)
        self.assertEqual(result["backend"], "gloo")
        self.assertEqual(result["world_size"], 2)
        self.assertEqual(result["hidden_width"], 4)
        for key in (
            "forward_max_error", "weight_gradient_max_error",
            "input_gradient_max_error", "post_step_max_error",
        ):
            self.assertLess(result[key], 1e-10, key)


if __name__ == "__main__":
    unittest.main()
