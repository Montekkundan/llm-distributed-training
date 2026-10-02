"""Two CPU ranks dispatch top-2 tokens to expert owners and return gradients.

Two exchange patterns share one routing rule and one serial reference:

* ``run_expert_parity`` (the default): rank 0 owns every token and rank 1 owns no
  token. Rank 0 sends the remote experts' rows to rank 1 with ``dist.send`` and
  receives results with ``dist.recv``. This is point-to-point traffic from a single
  source, not an all-to-all.
* ``run_all_to_all_parity`` (``--all-to-all``): both ranks own tokens (round-robin)
  and both own experts, and rows move with ``dist.all_to_all_single``, the
  collective that expert parallelism uses. The routing decision is replicated:
  every rank holds all logits as constants and applies the same global capacity
  rule, so no per-expert counts are communicated for it.

Both run two Gloo CPU ranks in float64. Neither measures bandwidth or time.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
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


@lru_cache(maxsize=None)
def _exchange_function():
    """Autograd wrapper for dist.all_to_all_single with explicit row counts."""
    import torch
    import torch.distributed as dist

    class Exchange(torch.autograd.Function):
        @staticmethod
        def forward(ctx, rows, send_counts, receive_counts):
            ctx.counts = (send_counts, receive_counts)
            received = rows.new_empty((sum(receive_counts), *rows.shape[1:]))
            dist.all_to_all_single(received, rows.contiguous(),
                                   output_split_sizes=list(receive_counts),
                                   input_split_sizes=list(send_counts))
            return received

        @staticmethod
        def backward(ctx, gradient):
            # Moving rows is a permutation, so its adjoint is the inverse move:
            # the same exchange with send and receive counts swapped.
            send_counts, receive_counts = ctx.counts
            returned = gradient.new_empty((sum(send_counts), *gradient.shape[1:]))
            dist.all_to_all_single(returned, gradient.contiguous(),
                                   output_split_sizes=list(send_counts),
                                   input_split_sizes=list(receive_counts))
            return returned, None, None

    return Exchange


def _all_to_all_worker(rank: int, rendezvous: str, report_path: str) -> None:
    import torch
    import torch.distributed as dist
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

        # Token t lives on rank t % 2 (3 tokens on rank 0, 2 on rank 1); expert e lives on rank e // 2.
        owned = list(range(rank, 5, 2))
        owned_index = torch.tensor(owned, dtype=torch.int64)
        experts_per_rank = 2
        local_inputs = reference_inputs.detach()[owned_index].clone().requires_grad_()
        local_logits = reference_logits.detach()[owned_index].clone().requires_grad_()
        local_weights = reference_weights.detach()[
            rank * experts_per_rank:(rank + 1) * experts_per_rank].clone().requires_grad_()

        # Replicated routing decision: all rows are constants except this rank's,
        # so the gate probabilities carry gradient only into local_logits.
        all_logits = reference_logits.detach().index_copy(0, owned_index, local_logits)
        accepted, probabilities, _ = route(all_logits, top_k=2, capacity=2)
        mine = sorted((item for item in accepted if item[0] in owned),
                      key=lambda item: item[2] // experts_per_rank)  # stable: token-major per destination
        send_counts = [sum(expert // experts_per_rank == destination for _, _, expert in mine)
                       for destination in range(2)]
        counts_in = torch.empty(2, dtype=torch.int64)
        dist.all_to_all_single(counts_in, torch.tensor(send_counts, dtype=torch.int64))
        receive_counts = counts_in.tolist()

        exchange = _exchange_function()
        local_rows = torch.tensor([owned.index(token) for token, _, _ in mine], dtype=torch.int64)
        dispatched = exchange.apply(local_inputs[local_rows], send_counts, receive_counts)
        expert_ids = torch.tensor([expert for _, _, expert in mine], dtype=torch.int64)
        dispatched_experts = torch.empty(sum(receive_counts), dtype=torch.int64)
        dist.all_to_all_single(dispatched_experts, expert_ids,
                               output_split_sizes=receive_counts, input_split_sizes=send_counts)
        owned_weights = local_weights[dispatched_experts - rank * experts_per_rank]
        expert_outputs = torch.bmm(dispatched[:, None, :], owned_weights)[:, 0, :]
        # Reverse exchange: rows come back in the order they were sent.
        returned = exchange.apply(expert_outputs, receive_counts, send_counts)

        gates = probabilities[torch.tensor([token for token, _, _ in mine], dtype=torch.int64),
                              torch.tensor([slot for _, slot, _ in mine], dtype=torch.int64)]
        combined = torch.zeros_like(local_inputs).index_add(0, local_rows, returned * gates[:, None])
        # Global mean over all 5 x 2 outputs, so the per-rank gradients add up to the serial ones.
        loss = (combined - targets[owned_index]).square().sum() / targets.numel()
        loss.backward()

        forward_error = _max_error(combined, reference_output[owned_index])
        gradient_error = max(
            _max_error(local_weights.grad, reference_weights.grad[
                rank * experts_per_rank:(rank + 1) * experts_per_rank]),
            _max_error(local_inputs.grad, reference_inputs.grad[owned_index]),
            _max_error(local_logits.grad, reference_logits.grad[owned_index]))
        with torch.no_grad():
            expert_span = slice(rank * experts_per_rank, (rank + 1) * experts_per_rank)
            for local, reference, rows in ((local_weights, reference_weights, expert_span),
                                           (local_inputs, reference_inputs, owned_index),
                                           (local_logits, reference_logits, owned_index)):
                local -= 0.05 * local.grad
                reference[rows] -= 0.05 * reference.grad[rows]
            post_step_error = max(_max_error(local_weights, reference_weights[expert_span]),
                                  _max_error(local_inputs, reference_inputs[owned_index]),
                                  _max_error(local_logits, reference_logits[owned_index]))
        errors = torch.tensor([forward_error, gradient_error, post_step_error], dtype=torch.float64)
        dist.all_reduce(errors, op=dist.ReduceOp.MAX)
        matrix = [torch.empty(2, dtype=torch.int64) for _ in range(2)]
        dist.all_gather(matrix, torch.tensor(send_counts, dtype=torch.int64))
        if rank == 0:
            Path(report_path).write_text(json.dumps({
                "backend": "gloo", "collective": "all_to_all_single", "world_size": 2,
                "top_k": 2, "capacity": 2, "expert_counts": counts,
                "dropped_assignments": 5 * 2 - len(expected_routes),
                "tokens_per_rank": [3, 2],
                "rows_sent": [row.tolist() for row in matrix],  # [source rank][destination rank]
                "forward_error": errors[0].item(),
                "gradient_error": errors[1].item(),
                "post_step_error": errors[2].item(),
            }, sort_keys=True))
    finally:
        dist.destroy_process_group()


def run_expert_parity() -> dict[str, float | int | str | list[int]]:
    """Check two expert owners against serial top-2/capacity-limited routing."""
    return _run(_worker)


def run_all_to_all_parity() -> dict[str, float | int | str | list]:
    """Same check with tokens on both ranks and rows moved by all_to_all_single."""
    return _run(_all_to_all_worker)


if __name__ == "__main__":
    print(json.dumps(run_all_to_all_parity() if "--all-to-all" in sys.argv[1:]
                     else run_expert_parity(), sort_keys=True))
