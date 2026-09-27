"""Two-rank CPU communication labs for pipeline and context parallelism."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory


def _max_error(actual, expected) -> float:
    return float((actual.detach() - expected.detach()).abs().max())


def _pipeline_worker(rank: int, rendezvous: str, report_path: str) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch import nn

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        torch.manual_seed(23)
        serial = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
        torch.manual_seed(23)
        complete = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
        stage = nn.Sequential(*list(complete.children())[:2]) if rank == 0 else complete[2]
        inputs = torch.tensor([[0.2, -0.8, 0.4], [1.1, 0.3, -0.5],
                               [-0.4, 0.7, 0.2], [0.9, -1.2, 0.1]], dtype=torch.float64)
        targets = torch.tensor([[0.1, -0.2], [0.7, 0.3], [-0.5, 0.4], [0.2, -0.8]],
                               dtype=torch.float64)
        reference_output = serial(inputs)
        F.mse_loss(reference_output, targets).backward()

        activations = []
        predictions = []
        for microbatch in range(2):
            if rank == 0:
                activation = stage(inputs[microbatch * 2:(microbatch + 1) * 2])
                activations.append(activation)
                dist.send(activation.detach().contiguous(), dst=1, tag=100 + microbatch)
            else:
                activation = torch.empty((2, 4), dtype=torch.float64)
                dist.recv(activation, src=0, tag=100 + microbatch)
                activation.requires_grad_()
                activations.append(activation)
                predictions.append(stage(activation))

        for microbatch in reversed(range(2)):
            if rank == 1:
                loss = F.mse_loss(predictions[microbatch],
                                  targets[microbatch * 2:(microbatch + 1) * 2]) / 2
                loss.backward()
                dist.send(activations[microbatch].grad.contiguous(), dst=0,
                          tag=200 + microbatch)
            else:
                gradient = torch.empty((2, 4), dtype=torch.float64)
                dist.recv(gradient, src=1, tag=200 + microbatch)
                activations[microbatch].backward(gradient)

        if rank == 0:
            forward_error = _max_error(torch.cat([value.detach() for value in activations]),
                                       serial[:2](inputs).detach())
            reference_parameters = list(serial[:2].parameters())
        else:
            forward_error = _max_error(torch.cat([value.detach() for value in predictions]),
                                       reference_output.detach())
            reference_parameters = list(serial[2].parameters())

        gradient_error = max(_max_error(parameter.grad, reference.grad)
                             for parameter, reference in zip(stage.parameters(), reference_parameters))
        with torch.no_grad():
            for parameter, reference in zip(stage.parameters(), reference_parameters):
                parameter -= 0.05 * parameter.grad
                reference -= 0.05 * reference.grad
        post_step_error = max(_max_error(parameter, reference)
                              for parameter, reference in zip(stage.parameters(), reference_parameters))
        errors = torch.tensor([forward_error, gradient_error, post_step_error], dtype=torch.float64)
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if rank == 0:
            Path(report_path).write_text(json.dumps({
                "backend": "gloo", "world_size": 2, "microbatches": 2,
                "forward_error": errors[0].item(),
                "gradient_error": errors[1].item(),
                "post_step_error": errors[2].item(),
            }, sort_keys=True))
    finally:
        dist.destroy_process_group()


def _attention(query, blocks, query_indices):
    import torch

    maximum = torch.full((len(query_indices), 1), -torch.inf, dtype=query.dtype)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros((len(query_indices), query.shape[-1]), dtype=query.dtype)
    for key, value, key_indices in blocks:
        if key_indices[0] > query_indices[-1]:
            continue
        scores = query @ key.T / query.shape[-1] ** 0.5
        allowed = query_indices[:, None] >= key_indices[None, :]
        scores = scores.masked_fill(~allowed, -torch.inf)
        next_maximum = torch.maximum(maximum, scores.max(dim=-1, keepdim=True).values)
        prior_scale = torch.where(torch.isfinite(maximum),
                                  torch.exp(maximum - next_maximum), 0.0)
        weights = torch.exp(scores - next_maximum)
        denominator = denominator * prior_scale + weights.sum(dim=-1, keepdim=True)
        numerator = numerator * prior_scale + weights @ value
        maximum = next_maximum
    return numerator / denominator


def _context_worker(rank: int, rendezvous: str, report_path: str) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        queries = torch.tensor([[0.2, -0.4], [0.8, 0.1], [-0.5, 1.0], [1.2, -0.7]],
                               dtype=torch.float64, requires_grad=True)
        keys = torch.tensor([[0.6, 0.3], [-0.4, 0.8], [0.5, -0.9], [0.7, 0.2]],
                            dtype=torch.float64, requires_grad=True)
        values = torch.tensor([[0.1, -0.3], [0.4, 0.9], [-0.8, 0.2], [0.5, -0.6]],
                              dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0.2, 0.1], [0.1, -0.3], [0.4, 0.2], [-0.2, 0.8]],
                               dtype=torch.float64)
        positions = torch.arange(4)
        reference_output = _attention(queries, [(keys, values, positions)], positions)
        F.mse_loss(reference_output, targets).backward()

        span = slice(rank * 2, (rank + 1) * 2)
        local_q = queries.detach()[span].clone().requires_grad_()
        local_k = keys.detach()[span].clone().requires_grad_()
        local_v = values.detach()[span].clone().requires_grad_()
        outgoing = torch.cat((local_k.detach(), local_v.detach()), dim=-1).contiguous()
        incoming = torch.empty_like(outgoing)
        if rank == 0:
            dist.send(outgoing, dst=1, tag=300)
            dist.recv(incoming, src=1, tag=300)
        else:
            dist.recv(incoming, src=0, tag=300)
            dist.send(outgoing, dst=0, tag=300)
        remote_k = incoming[:, :2].clone().requires_grad_()
        remote_v = incoming[:, 2:].clone().requires_grad_()
        local_positions = torch.arange(rank * 2, rank * 2 + 2)
        remote_positions = torch.arange((1 - rank) * 2, (1 - rank) * 2 + 2)
        output = _attention(local_q, [(local_k, local_v, local_positions),
                                       (remote_k, remote_v, remote_positions)],
                            local_positions)
        F.mse_loss(output, targets[span]).div(2).backward()
        remote_gradients = torch.cat((
            remote_k.grad if remote_k.grad is not None else torch.zeros_like(remote_k),
            remote_v.grad if remote_v.grad is not None else torch.zeros_like(remote_v),
        ), dim=-1).contiguous()
        received_gradients = torch.empty_like(remote_gradients)
        if rank == 0:
            dist.send(remote_gradients, dst=1, tag=400)
            dist.recv(received_gradients, src=1, tag=400)
        else:
            dist.recv(received_gradients, src=0, tag=400)
            dist.send(remote_gradients, dst=0, tag=400)
        local_k.grad += received_gradients[:, :2]
        local_v.grad += received_gradients[:, 2:]

        forward_error = _max_error(output.detach(), reference_output.detach()[span])
        gradient_error = max(_max_error(local.grad, reference.grad[span])
                             for local, reference in ((local_q, queries),
                                                      (local_k, keys), (local_v, values)))
        with torch.no_grad():
            for local, reference in ((local_q, queries), (local_k, keys), (local_v, values)):
                local -= 0.05 * local.grad
                reference -= 0.05 * reference.grad
        post_step_error = max(_max_error(local, reference[span])
                              for local, reference in ((local_q, queries),
                                                       (local_k, keys), (local_v, values)))
        errors = torch.tensor([forward_error, gradient_error, post_step_error], dtype=torch.float64)
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if rank == 0:
            Path(report_path).write_text(json.dumps({
                "backend": "gloo", "world_size": 2, "sequence_length": 4,
                "forward_error": errors[0].item(),
                "gradient_error": errors[1].item(),
                "post_step_error": errors[2].item(),
            }, sort_keys=True))
    finally:
        dist.destroy_process_group()


def _run(worker) -> dict[str, float | int | str]:
    try:
        import torch.distributed as dist
        import torch.multiprocessing as mp
    except ImportError as error:
        raise RuntimeError("Install the optional torch dependency to run this lab") from error
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("This PyTorch build needs the Gloo distributed backend")
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with TemporaryDirectory(prefix="process-parallel-") as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        report_path = str(Path(directory) / "report.json")
        mp.spawn(worker, args=(rendezvous, report_path), nprocs=2, join=True)
        return json.loads(Path(report_path).read_text())


def run_pipeline_parity() -> dict[str, float | int | str]:
    """Check two physical stages and two fill-drain microbatches."""
    return _run(_pipeline_worker)


def run_context_parity() -> dict[str, float | int | str]:
    """Check exchanged K/V blocks and returned gradients against serial attention."""
    return _run(_context_worker)


if __name__ == "__main__":
    print(json.dumps({"pipeline": run_pipeline_parity(),
                      "context": run_context_parity()}, sort_keys=True))
