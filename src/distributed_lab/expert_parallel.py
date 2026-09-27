"""Two CPU ranks dispatch top-2 tokens to expert owners and return gradients."""

from __future__ import annotations

import json
from pathlib import Path

from .process_parallel import _max_error, _run


def route(logits, *, top_k: int, capacity: int):
    """Accept token-major top-k requests until each expert reaches capacity.

    Surviving gate weights use a softmax over the original top-k scores and are
    not renormalized when a request is dropped.
    """
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have shape [nonempty tokens, nonempty experts]")
    if (type(top_k) is not int or not 1 <= top_k <= logits.shape[1]
            or type(capacity) is not int or capacity < 1):
        raise ValueError("top_k and capacity must be valid positive integers")
    selected = logits.topk(top_k, dim=-1)
    probabilities = selected.values.softmax(dim=-1)
    counts = [0] * logits.shape[1]
    accepted = []
    for token, experts in enumerate(selected.indices.tolist()):
        for slot, expert in enumerate(experts):
            if counts[expert] < capacity:
                accepted.append((token, slot, expert))
                counts[expert] += 1
    return accepted, probabilities, tuple(counts)


def _serial_output(inputs, logits, weights, targets):
    import torch
    import torch.nn.functional as F

    accepted, probabilities, counts = route(logits, top_k=2, capacity=2)
    outputs = torch.stack([inputs[token] @ weights[expert]
                           for token, _, expert in accepted])
    gates = torch.stack([probabilities[token, slot]
                         for token, slot, _ in accepted])
    indices = torch.tensor([token for token, _, _ in accepted])
    combined = torch.zeros_like(inputs).index_add(0, indices, outputs * gates[:, None])
    loss = F.mse_loss(combined, targets)
    return combined, loss, accepted, counts


def _worker(rank: int, rendezvous: str, report_path: str) -> None:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from datetime import timedelta

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        inputs_data = [[0.2, -0.8], [1.1, 0.3], [-0.4, 0.7],
                       [0.9, -1.2], [0.6, 0.5]]
        logits_data = [[2., 1., 0., -1.], [2., 1., 0., -1.],
                       [2., 0., 1., -1.], [2., -1., 0., 1.],
                       [-1., 0., 2., 1.]]
        weight_data = [[[0.2, -0.3], [0.4, 0.7]],
                       [[-0.5, 0.1], [0.8, 0.2]],
                       [[0.3, 0.9], [-0.2, 0.5]],
                       [[-0.7, 0.4], [0.6, -0.1]]]
        targets = torch.tensor([[0.1, -0.2], [0.7, 0.3], [-0.5, 0.4],
                                [0.2, -0.8], [0.3, 0.1]], dtype=torch.float64)
        reference_inputs = torch.tensor(inputs_data, dtype=torch.float64, requires_grad=True)
        reference_logits = torch.tensor(logits_data, dtype=torch.float64, requires_grad=True)
        reference_weights = torch.tensor(weight_data, dtype=torch.float64, requires_grad=True)
        reference_output, reference_loss, expected_routes, counts = _serial_output(
            reference_inputs, reference_logits, reference_weights, targets)
        reference_loss.backward()

        local_weights = reference_weights.detach()[rank * 2:(rank + 1) * 2].clone().requires_grad_()
        if rank == 0:
            inputs = reference_inputs.detach().clone().requires_grad_()
            logits = reference_logits.detach().clone().requires_grad_()
            accepted, probabilities, _ = route(logits, top_k=2, capacity=2)
            local_routes = [item for item in accepted if item[2] < 2]
            remote_routes = [item for item in accepted if item[2] >= 2]
            local_inputs = torch.stack([inputs[token] for token, _, _ in local_routes])
            local_outputs = torch.stack([local_inputs[i] @ local_weights[expert]
                                         for i, (_, _, expert) in enumerate(local_routes)])
            remote_inputs = torch.stack([inputs[token] for token, _, _ in remote_routes])
            remote_metadata = torch.tensor([[token, expert] for token, _, expert in remote_routes],
                                           dtype=torch.int64)
            remote_count = torch.tensor([len(remote_routes)], dtype=torch.int64)
            dist.send(remote_count, dst=1, tag=500)
            dist.send(remote_metadata, dst=1, tag=501)
            dist.send(remote_inputs.detach().contiguous(), dst=1, tag=502)
            remote_outputs = torch.empty_like(remote_inputs)
            dist.recv(remote_outputs, src=1, tag=503)
            remote_outputs.requires_grad_()

            routes = local_routes + remote_routes
            gate_weights = torch.stack([probabilities[token, slot] for token, slot, _ in routes])
            outputs = torch.cat((local_outputs, remote_outputs), dim=0)
            token_ids = torch.tensor([token for token, _, _ in routes])
            combined = torch.zeros_like(inputs).index_add(
                0, token_ids, outputs * gate_weights[:, None])
            F.mse_loss(combined, targets).backward()
            dist.send(remote_outputs.grad.contiguous(), dst=1, tag=504)
            remote_input_gradients = torch.empty_like(remote_inputs)
            dist.recv(remote_input_gradients, src=1, tag=505)
            remote_inputs.backward(remote_input_gradients)

            forward_error = _max_error(combined, reference_output)
            gradient_error = max(
                _max_error(local_weights.grad, reference_weights.grad[:2]),
                _max_error(inputs.grad, reference_inputs.grad),
                _max_error(logits.grad, reference_logits.grad))
        else:
            remote_count = torch.empty(1, dtype=torch.int64)
            dist.recv(remote_count, src=0, tag=500)
            remote_metadata = torch.empty((int(remote_count.item()), 2), dtype=torch.int64)
            dist.recv(remote_metadata, src=0, tag=501)
            remote_inputs = torch.empty((len(remote_metadata), 2), dtype=torch.float64)
            dist.recv(remote_inputs, src=0, tag=502)
            remote_inputs.requires_grad_()
            remote_outputs = torch.stack([
                remote_inputs[i] @ local_weights[int(expert.item()) - 2]
                for i, (_, expert) in enumerate(remote_metadata)
            ])
            dist.send(remote_outputs.detach().contiguous(), dst=0, tag=503)
            remote_output_gradients = torch.empty_like(remote_outputs)
            dist.recv(remote_output_gradients, src=0, tag=504)
            remote_outputs.backward(remote_output_gradients)
            dist.send(remote_inputs.grad.contiguous(), dst=0, tag=505)

            forward_error = 0.0
            gradient_error = _max_error(local_weights.grad, reference_weights.grad[2:])

        with torch.no_grad():
            local_weights -= 0.05 * local_weights.grad
            reference_weights[rank * 2:(rank + 1) * 2] -= (
                0.05 * reference_weights.grad[rank * 2:(rank + 1) * 2])
            post_step_error = _max_error(
                local_weights, reference_weights[rank * 2:(rank + 1) * 2])
            if rank == 0:
                inputs -= 0.05 * inputs.grad
                logits -= 0.05 * logits.grad
                reference_inputs -= 0.05 * reference_inputs.grad
                reference_logits -= 0.05 * reference_logits.grad
                post_step_error = max(post_step_error,
                                      _max_error(inputs, reference_inputs),
                                      _max_error(logits, reference_logits))
        errors = torch.tensor([forward_error, gradient_error, post_step_error], dtype=torch.float64)
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        if rank == 0:
            Path(report_path).write_text(json.dumps({
                "backend": "gloo", "world_size": 2, "top_k": 2, "capacity": 2,
                "expert_counts": counts,
                "dropped_assignments": 5 * 2 - len(expected_routes),
                "forward_error": errors[0].item(),
                "gradient_error": errors[1].item(),
                "post_step_error": errors[2].item(),
            }, sort_keys=True))
    finally:
        dist.destroy_process_group()


def run_expert_parity() -> dict[str, float | int | str | list[int]]:
    """Check two expert owners against serial top-2/capacity-limited routing."""
    return _run(_worker)


if __name__ == "__main__":
    print(json.dumps(run_expert_parity(), sort_keys=True))
