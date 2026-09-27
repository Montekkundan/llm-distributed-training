"""A real two-process, CPU/Gloo tensor-parallel FFN correctness lab.

Run ``python -m distributed_lab.torch_distributed`` with PyTorch installed.
The standard-library contract exercises remain usable without PyTorch.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _check_close(actual: Any, expected: Any, label: str) -> float:
    import torch

    error = float((actual - expected).abs().max())
    if not torch.allclose(actual, expected, rtol=1e-10, atol=1e-11):
        raise AssertionError(f"{label} differs from the serial reference: max error {error}")
    return error


def _worker(rank: int, rendezvous: str, report_path: str, hidden_width: int) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional

    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        # The same input and target are replicated; only the hidden weights are sharded.
        x_values = torch.tensor([[0.2, -1.0, 0.5], [1.3, 0.4, -0.7]], dtype=torch.float64)
        target = torch.tensor([[0.1, -0.2], [0.6, 0.3]], dtype=torch.float64)
        first_values = torch.arange(3 * hidden_width, dtype=torch.float64).reshape(3, hidden_width) / 13 - 0.4
        second_values = torch.arange(hidden_width * 2, dtype=torch.float64).reshape(hidden_width, 2) / 11 - 0.2
        shard = slice(rank * (hidden_width // 2), (rank + 1) * (hidden_width // 2))

        serial_x = x_values.clone().requires_grad_()
        serial_first = first_values.clone().requires_grad_()
        serial_second = second_values.clone().requires_grad_()
        serial_output = functional.gelu(serial_x @ serial_first) @ serial_second
        serial_loss = 0.5 * (serial_output - target).square().mean()
        serial_loss.backward()

        local_x = x_values.clone().requires_grad_()
        local_first = first_values[:, shard].clone().requires_grad_()
        local_second = second_values[shard, :].clone().requires_grad_()
        local_partial = functional.gelu(local_x @ local_first) @ local_second

        # Gloo all-reduce creates the complete output on both ranks. Backpropagate
        # through *this rank's* partial output with the shared global loss gradient.
        # Ordinary dist.all_reduce on a detached buffer is not an autograd operation.
        global_output = local_partial.detach().clone()
        dist.all_reduce(global_output, op=dist.ReduceOp.SUM)
        output_error = _check_close(global_output, serial_output.detach(), "forward output")
        output_gradient = (global_output - target) / global_output.numel()
        local_partial.backward(output_gradient)
        first_error = _check_close(local_first.grad, serial_first.grad[:, shard], "first weight gradient")
        second_error = _check_close(local_second.grad, serial_second.grad[shard, :], "second weight gradient")
        dist.all_reduce(local_x.grad, op=dist.ReduceOp.SUM)
        input_error = _check_close(local_x.grad, serial_x.grad, "input gradient")

        learning_rate = 0.05
        with torch.no_grad():
            local_first -= learning_rate * local_first.grad
            local_second -= learning_rate * local_second.grad
            serial_first -= learning_rate * serial_first.grad
            serial_second -= learning_rate * serial_second.grad
            _check_close(local_first, serial_first[:, shard], "updated first weight")
            _check_close(local_second, serial_second[shard, :], "updated second weight")
            next_partial = functional.gelu(x_values @ local_first) @ local_second
            next_output = next_partial.clone()
            dist.all_reduce(next_output, op=dist.ReduceOp.SUM)
            next_reference = functional.gelu(x_values @ serial_first) @ serial_second
            step_error = _check_close(next_output, next_reference, "post-step output")

        # Each rank must complete every assertion before rank zero reports success.
        dist.barrier()
        if rank == 0:
            Path(report_path).write_text(json.dumps({
                "backend": dist.get_backend(),
                "world_size": dist.get_world_size(),
                "hidden_width": hidden_width,
                "forward_max_error": output_error,
                "weight_gradient_max_error": max(first_error, second_error),
                "input_gradient_max_error": input_error,
                "post_step_max_error": step_error,
            }, sort_keys=True))
    finally:
        dist.destroy_process_group()


def run_two_rank_parity(hidden_width: int = 4) -> dict[str, float | int | str]:
    """Spawn two local Gloo ranks and compare one FFN update with serial autograd."""
    if type(hidden_width) is not int or hidden_width < 2 or hidden_width % 2:
        raise ValueError("hidden_width must be an even integer of at least 2")
    try:
        import torch.distributed as dist
        import torch.multiprocessing as mp
    except ImportError as error:
        raise RuntimeError("Install the optional torch dependency to run this lab") from error
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("This PyTorch build needs the Gloo distributed backend")
    if sys.platform == "darwin":
        # macOS hostname discovery can select an unusable interface for Gloo.
        # Respect an explicit choice; otherwise bind the local two-rank lab to loopback.
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with TemporaryDirectory(prefix="distributed-lab-") as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        report_path = str(Path(directory) / "report.json")
        mp.spawn(_worker, args=(rendezvous, report_path, hidden_width), nprocs=2, join=True)
        return json.loads(Path(report_path).read_text())


def main() -> None:
    print(json.dumps(run_two_rank_parity(), sort_keys=True))


if __name__ == "__main__":
    main()
