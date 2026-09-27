"""Two CPU/Gloo ranks train a tiny causal decoder against a serial reference."""

from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from .torch_distributed import _check_close


class TinyCausalDecoder(nn.Module):
    """One pre-norm attention/MLP block with next-token logits and no dropout."""

    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Embedding(17, 8)
        self.position = nn.Embedding(8, 8)
        self.attention_norm = nn.LayerNorm(8)
        self.qkv = nn.Linear(8, 24)
        self.attention_out = nn.Linear(8, 8)
        self.mlp_norm = nn.LayerNorm(8)
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        self.final_norm = nn.LayerNorm(8)
        self.head = nn.Linear(8, 17, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        if length > 8:
            raise ValueError("sequence length exceeds the eight-position toy context")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)
        query, key, value = self.qkv(self.attention_norm(hidden)).chunk(3, dim=-1)
        scores = query @ key.transpose(-1, -2) / (query.shape[-1] ** 0.5)
        future = torch.ones(length, length, dtype=torch.bool, device=tokens.device).triu(1)
        attention = scores.masked_fill(future, -torch.inf).softmax(dim=-1) @ value
        hidden = hidden + self.attention_out(attention)
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return self.head(self.final_norm(hidden))


def _examples() -> tuple[torch.Tensor, torch.Tensor]:
    sequences = torch.tensor([
        [1, 2, 3, 4, 5, 6], [2, 1, 4, 3, 6, 5],
        [7, 8, 9, 3, 2, 1], [9, 7, 1, 8, 4, 2],
    ])
    return sequences[:, :-1], sequences[:, 1:]


def _loss(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def _worker(rank: int, rendezvous: str, report_path: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        torch.manual_seed(19)
        serial = TinyCausalDecoder().double()
        torch.manual_seed(19)
        replica = DistributedDataParallel(TinyCausalDecoder().double())
        inputs, targets = _examples()
        local_inputs = inputs[rank * 2:(rank + 1) * 2]
        local_targets = targets[rank * 2:(rank + 1) * 2]

        serial_loss = _loss(serial, inputs, targets)
        serial_loss.backward()
        local_loss = _loss(replica, local_inputs, local_targets)
        local_loss.backward()  # DDP all-reduces and averages each parameter gradient.
        mean_loss = local_loss.detach().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
        mean_loss /= 2
        loss_error = _check_close(mean_loss, serial_loss.detach(), "mean shard loss")

        # The local mean alone is generally wrong: the synchronized average matters.
        torch.manual_seed(19)
        unsynchronized = TinyCausalDecoder().double()
        _loss(unsynchronized, local_inputs, local_targets).backward()
        local_only_error = max(float((a.grad - b.grad).abs().max())
                               for a, b in zip(unsynchronized.parameters(), serial.parameters()))

        parameter_error = max(_check_close(a.detach(), b.detach(), "initial parameter")
                              for a, b in zip(replica.module.parameters(), serial.parameters()))
        gradient_error = max(_check_close(a.grad, b.grad, "averaged gradient")
                             for a, b in zip(replica.module.parameters(), serial.parameters()))
        serial_optimizer = torch.optim.SGD(serial.parameters(), lr=0.05)
        replica_optimizer = torch.optim.SGD(replica.parameters(), lr=0.05)
        serial_optimizer.step()
        replica_optimizer.step()
        step_error = max(_check_close(a.detach(), b.detach(), "updated parameter")
                         for a, b in zip(replica.module.parameters(), serial.parameters()))
        serial_next_loss = _loss(serial, inputs, targets)
        shard_next_loss = _loss(replica, local_inputs, local_targets).detach()
        dist.all_reduce(shard_next_loss, op=dist.ReduceOp.SUM)
        shard_next_loss /= 2
        next_loss_error = _check_close(shard_next_loss, serial_next_loss.detach(), "post-step loss")
        dist.barrier()
        if rank == 0:
            Path(report_path).write_text(json.dumps({
                "backend": dist.get_backend(), "world_size": dist.get_world_size(),
                "examples_per_rank": 2, "loss_max_error": loss_error,
                "gradient_max_error": gradient_error,
                "parameter_max_error": parameter_error,
                "post_step_max_error": step_error,
                "post_step_loss_max_error": next_loss_error,
                "local_only_gradient_max_error": local_only_error,
            }, sort_keys=True))
    finally:
        dist.destroy_process_group()


def run_two_rank_decoder_parity() -> dict[str, float | int | str]:
    """Compare two real DDP ranks with one full-batch decoder SGD update."""
    if not dist.is_available() or not dist.is_gloo_available():
        raise RuntimeError("This PyTorch build needs the Gloo distributed backend")
    if sys.platform == "darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    with TemporaryDirectory(prefix="ddp-decoder-") as directory:
        rendezvous = (Path(directory) / "rendezvous").as_uri()
        report_path = str(Path(directory) / "report.json")
        mp.spawn(_worker, args=(rendezvous, report_path), nprocs=2, join=True)
        return json.loads(Path(report_path).read_text())


if __name__ == "__main__":
    print(json.dumps(run_two_rank_decoder_parity(), sort_keys=True))
