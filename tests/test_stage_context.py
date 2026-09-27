"""Two-rank pipeline and context communication parity checks."""

from __future__ import annotations

import unittest


def _has_gloo() -> bool:
    try:
        import torch.distributed as dist
    except ImportError:
        return False
    return dist.is_available() and dist.is_gloo_available()


@unittest.skipUnless(_has_gloo(), "optional PyTorch Gloo backend is unavailable")
class ProcessParallelTests(unittest.TestCase):
    def test_pipeline_forward_gradients_and_step_match_serial(self) -> None:
        from distributed_lab.process_parallel import run_pipeline_parity

        result = run_pipeline_parity()
        self.assertEqual((result["backend"], result["world_size"]), ("gloo", 2))
        for key in ("forward_error", "gradient_error", "post_step_error"):
            self.assertLess(result[key], 1e-10, key)

    def test_context_forward_gradients_and_step_match_serial(self) -> None:
        from distributed_lab.process_parallel import run_context_parity

        result = run_context_parity()
        self.assertEqual((result["backend"], result["world_size"]), ("gloo", 2))
        for key in ("forward_error", "gradient_error", "post_step_error"):
            self.assertLess(result[key], 1e-10, key)


if __name__ == "__main__":
    unittest.main()
