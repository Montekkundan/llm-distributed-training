"""Real CPU/Gloo data-parallel checks for a tiny causal language model."""

from __future__ import annotations

import unittest


def _has_gloo() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return dist.is_available() and dist.is_gloo_available()


@unittest.skipUnless(_has_gloo(), "optional PyTorch Gloo backend is unavailable")
class CausalDecoderDDPTests(unittest.TestCase):
    def test_two_ranks_match_full_batch_gradients_and_step(self) -> None:
        from distributed_lab.ddp_decoder import run_two_rank_decoder_parity

        result = run_two_rank_decoder_parity()
        self.assertEqual(result["backend"], "gloo")
        self.assertEqual(result["world_size"], 2)
        self.assertEqual(result["examples_per_rank"], 2)
        for key in (
            "loss_max_error", "gradient_max_error", "parameter_max_error",
            "post_step_max_error", "post_step_loss_max_error",
        ):
            self.assertLess(result[key], 1e-10, key)
        # A single rank's local mean is not the full-batch mean.
        self.assertGreater(result["local_only_gradient_max_error"], 1e-6)

    def test_future_tokens_cannot_change_prefix_logits(self) -> None:
        import torch

        from distributed_lab.ddp_decoder import TinyCausalDecoder

        torch.manual_seed(19)
        model = TinyCausalDecoder().double()
        first = torch.tensor([[1, 2, 3, 4, 5]])
        changed_future = torch.tensor([[1, 2, 3, 7, 8]])
        torch.testing.assert_close(model(first)[:, :3], model(changed_future)[:, :3], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
