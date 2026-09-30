"""CPU parity and failure tests for supplementary distributed contracts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from distributed_lab.schedules import Scheduled, activation_peaks, one_f_one_b, split_backward, validate
from distributed_lab.state import zero_ledger

try:
    import torch
except ImportError:
    torch = None


class ScheduleTests(unittest.TestCase):
    def test_1f1b_dependency_and_liveness(self):
        events = one_f_one_b(3, 6)
        self.assertEqual(max(event.end for event in events), 16)
        self.assertEqual(activation_peaks(events, 3), [3, 2, 1])
        self.assertTrue(validate(events, 3, 6, split=False))

    def test_delayed_weight_backward_retains_inputs(self):
        events = split_backward(3, 6)
        self.assertTrue(validate(events, 3, 6, split=True))
        peaks = activation_peaks(events, 3, split=True)
        self.assertTrue(all(peak >= 1 for peak in peaks))
        self.assertTrue(any(event.kind == "W" for event in events))
        # Compare equal F=1, complete backward=2, split input/weight backward=1+1.
        baseline = one_f_one_b(3, 6, backward_ticks=2)
        self.assertLessEqual(max(event.end for event in events), max(event.end for event in baseline))

    def test_missing_operation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate(one_f_one_b(2, 3)[:-1], 2, 3, split=False)
        with self.assertRaises(ValueError):
            validate((Scheduled(0, 0, "F", 0, 1), Scheduled(0, 0, "X", 1, 2)), 1, 1, split=False)
        with self.assertRaises(ValueError):
            validate((Scheduled(0, 0, "F", 0, 1), Scheduled(2, 0, "B", 1, 2)), 1, 1, split=False)

    def test_zero_ledger(self):
        self.assertEqual(zero_ledger(500_000_000, 4),
                         (9e9, 4.5e9, 3e9, 2.25e9))


@unittest.skipIf(torch is None, "optional PyTorch is unavailable")
class ReferenceParityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)

    def tensor(self, *shape):
        return torch.randn(*shape, dtype=torch.float64, requires_grad=True)

    def same_function_and_gradients(self, serial, partitioned, parameters):
        torch.testing.assert_close(serial, partitioned, rtol=1e-10, atol=1e-11)
        serial_grad = torch.autograd.grad(serial.square().mean(), parameters, retain_graph=True)
        parallel_grad = torch.autograd.grad(partitioned.square().mean(), parameters)
        for left, right in zip(serial_grad, parallel_grad):
            torch.testing.assert_close(left, right, rtol=1e-10, atol=1e-11)

    def test_zero_optimizer_ownership(self):
        from distributed_lab.state import adam_step, sharded_adam_step
        parameter, gradient, moment, variance = (self.tensor(7).detach() for _ in range(4))
        variance = variance.square()
        serial = adam_step(parameter, gradient, moment, variance, 4)
        sharded = sharded_adam_step(parameter, gradient, moment, variance, 4, 3)
        for left, right in zip(serial, sharded):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_delayed_weight_backward_and_version_guard(self):
        x, first, second = (torch.tensor(value, dtype=torch.float64, requires_grad=True)
                             for value in (2., .7, -.4))
        target = torch.tensor(.3, dtype=torch.float64)
        output = second * first * x
        expected = torch.autograd.grad((output - target).square() / 2, (x, first, second))
        output_adjoint = output.detach() - target
        input_adjoint = second.detach() * output_adjoint
        # Upstream input backward runs before downstream weight backward.
        actual = (first.detach() * input_adjoint,
                  x.detach() * input_adjoint, (first.detach() * x.detach()) * output_adjoint)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=1e-14, atol=1e-15)
        early_update_adjoint = (second.detach() - .05 * expected[2]) * output_adjoint
        self.assertNotEqual(float(early_update_adjoint), float(input_adjoint))

    def test_profiler_smoke_and_fp32_audit(self):
        from distributed_lab.profiling import audit_precision, run
        report = run(device="cpu", width=8, tokens=4, layers=2, segments=2, steps=2)
        self.assertEqual(report["shape"], [2, 4, 8])
        self.assertIsNone(report["peak_allocated_bytes"])
        self.assertGreater(report["median_step_seconds"], 0)
        audit = audit_precision(device="cpu", precision="fp32")
        self.assertEqual(audit["output_max_absolute_error"], 0)
        self.assertEqual(audit["gradient_max_absolute_error"], 0)
        self.assertTrue(audit["gradients_finite"])

    def test_fsdp_gather_and_owned_gradients(self):
        from distributed_lab.state import fsdp_reference
        x, weight, target = self.tensor(3, 4), self.tensor(4, 6), self.tensor(3, 6).detach()
        shards = [weight[:, :2].detach().clone().requires_grad_(),
                  weight[:, 2:].detach().clone().requires_grad_()]
        output, gradients = fsdp_reference(x, shards, target)
        expected = x @ weight
        full_gradient, = torch.autograd.grad((expected - target).square().mean(), weight)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.cat(gradients, dim=1), full_gradient, rtol=0, atol=0)

    def test_gqa_head_and_context_partition(self):
        from distributed_lab.parallel_contracts import attention, head_parallel_attention
        q, k, v, out = self.tensor(6, 4, 2), self.tensor(6, 2, 2), self.tensor(6, 2, 2), self.tensor(8, 3)
        serial = attention(q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1)).flatten(1) @ out
        partitioned = head_parallel_attention(q, k, v, out, 4, context_ranks=3)
        self.same_function_and_gradients(serial, partitioned, (q, k, v, out))

    def test_mla_projection_head_partition(self):
        from distributed_lab.parallel_contracts import mla
        q, latent, key, value, out = (self.tensor(*shape) for shape in
                                      ((5, 4, 2), (5, 3), (3, 4, 2), (3, 4, 2), (8, 3)))
        self.same_function_and_gradients(mla(q, latent, key, value, out, 1),
                                         mla(q, latent, key, value, out, 2),
                                         (q, latent, key, value, out))

    def test_expert_unequal_hidden_partitions(self):
        from distributed_lab.parallel_contracts import expert_tensor_parallel
        x, gate, up, down = self.tensor(3, 4), self.tensor(4, 6), self.tensor(4, 6), self.tensor(6, 4)
        serial = (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down
        self.same_function_and_gradients(serial, expert_tensor_parallel(x, gate, up, down, [1, 2, 3]),
                                         (x, gate, up, down))

    def test_joint_dense_ownership_reference(self):
        from distributed_lab.parallel_contracts import joint_reference, mesh_groups
        shapes = ((4, 6, 3), (3, 4, 2), (3, 2, 2), (3, 2, 2), (8, 3), (3, 6), (3, 6), (6, 3))
        parameters = tuple(self.tensor(*shape) for shape in shapes)
        serial = joint_reference(*parameters, dp=1, tp=1, cp=1)
        partitioned = joint_reference(*parameters, dp=2, tp=2, cp=3)
        self.same_function_and_gradients(serial, partitioned, parameters)
        mesh = mesh_groups(dp=2, tp=2, pp=2, cp=2, ep=1,
                           layers=24, heads=32, context=8192, experts=1)
        self.assertEqual(mesh["world_size"], 16)
        self.assertEqual(mesh["groups"]["tp"][0], [0, 4])
        with self.assertRaises(ValueError):
            mesh_groups(dp=2, tp=4, pp=4, cp=1, ep=1,
                        layers=30, heads=40, context=8192, experts=1)

    def test_atomic_sharded_resume_and_incomplete_rejection(self):
        from distributed_lab.checkpoint import load_checkpoint, load_latest, save_checkpoint
        from distributed_lab.state import sharded_adam_step
        def start():
            return self.tensor(6).detach(), torch.zeros(6, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)
        def advance(values, step):
            noise = torch.randn(6, dtype=torch.float64)
            gradient = 2 * (values[0] - (step + noise * .01))
            return sharded_adam_step(values[0], gradient, values[1], values[2], step + 1, 2)
        torch.manual_seed(43)
        full = start()
        for step in range(6):
            full = advance(full, step)
        torch.manual_seed(43)
        partial = start()
        for step in range(3):
            partial = advance(partial, step)
        with TemporaryDirectory() as root:
            shards = [{"model": partial[0][rank * 3:(rank + 1) * 3],
                       "optimizer": {"moment": partial[1][rank * 3:(rank + 1) * 3],
                                     "variance": partial[2][rank * 3:(rank + 1) * 3], "step": 3},
                       "rng": torch.get_rng_state(), "cursor": 3} for rank in range(2)]
            saved = save_checkpoint(root, step=3, shards=shards,
                                    metadata={"model_version": "toy-v1", "data_version": "generated-v1",
                                              "tokenizer_version": "none", "mesh": {"dp": 2}})
            manifest, restored = load_checkpoint(saved)
            self.assertEqual(manifest["step"], 3)
            resumed = (torch.cat([shard["model"] for shard in restored]),
                       torch.cat([shard["optimizer"]["moment"] for shard in restored]),
                       torch.cat([shard["optimizer"]["variance"] for shard in restored]))
            torch.set_rng_state(restored[0]["rng"])
            for step in range(restored[0]["cursor"], 6):
                resumed = advance(resumed, step)
            for left, right in zip(full, resumed):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            incomplete = Path(root) / "step-00000004"
            incomplete.mkdir()
            (incomplete / "rank-0000.pt").write_bytes(b"unfinished")
            with self.assertRaises(ValueError):
                load_checkpoint(incomplete)
            self.assertEqual(load_latest(root)[0]["step"], 3)
            (saved / "rank-0001.pt").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(saved)


if __name__ == "__main__":
    unittest.main()
