"""Autograd reference compositions, with ownership separate from communication."""

from itertools import product
from math import prod

from .partitions import balanced_ranges


def attention(query, key, value, *, query_positions=None):
    import torch
    # [tokens,heads,width]; keys/values may have replicated GQA heads.
    positions = torch.arange(query.shape[0]) if query_positions is None else query_positions
    scores = torch.einsum("thd,shd->hts", query, key) / query.shape[-1] ** 0.5
    future = torch.arange(key.shape[0])[None, :] > positions[:, None]
    probabilities = scores.masked_fill(future[None, :, :], -torch.inf).softmax(dim=-1)
    return torch.einsum("hts,shd->thd", probabilities, value)


def head_parallel_attention(query, key, value, out_weight, ranks, *, context_ranks=1):
    """CP query owners evaluate TP head slices and sum output-projection partials."""
    import torch
    tokens, heads, width = query.shape
    if heads % key.shape[1] or key.shape != value.shape:
        raise ValueError("GQA requires a whole query-head group per KV head")
    kv_for_query = torch.arange(heads) // (heads // key.shape[1])
    expanded_k, expanded_v = key[:, kv_for_query], value[:, kv_for_query]
    outputs = []
    for context in balanced_ranges(tokens, context_ranks, require_nonempty=True):
        positions = torch.tensor(list(context))
        pieces = []
        for span in balanced_ranges(heads, ranks, require_nonempty=True):
            head_ids = list(span)
            local = attention(query[positions][:, head_ids], expanded_k[:, head_ids],
                              expanded_v[:, head_ids], query_positions=positions)
            start, end = span.start * width, span.stop * width
            pieces.append(local.flatten(1) @ out_weight[start:end])
        outputs.append(torch.stack(pieces).sum(dim=0))
    return torch.cat(outputs)


def mla(query, latent, key_up, value_up, out_weight, ranks):
    """Shared compressed latent; reconstructed non-positional K/V split by heads.

    This isolates the MLA non-positional projection/TP identity. Decoupled RoPE,
    cache absorption, and communication are intentionally separate contracts.
    """
    import torch
    key = torch.einsum("tc,chd->thd", latent, key_up)
    value = torch.einsum("tc,chd->thd", latent, value_up)
    return head_parallel_attention(query, key, value, out_weight, ranks)


def expert_tensor_parallel(inputs, gate_weight, up_weight, down_weight, widths):
    """Unequal hidden-channel splits of one SwiGLU expert, summed before combine."""
    import torch
    from torch.nn.functional import silu
    if any(width < 1 for width in widths) or sum(widths) != gate_weight.shape[1]:
        raise ValueError("hidden partitions must cover each channel once")
    start = 0
    pieces = []
    for width in widths:
        span = slice(start, start + width)
        local = silu(inputs @ gate_weight[:, span]) * (inputs @ up_weight[:, span])
        pieces.append(local @ down_weight[span])
        start += width
    return torch.stack(pieces).sum(dim=0)


def mesh_groups(*, dp, tp, pp, cp, ep, layers, heads, context, experts):
    axes = (dp, tp, pp, cp, ep)
    if any(type(axis) is not int or axis < 1 for axis in axes):
        raise ValueError("positive integer mesh axes required")
    if layers % pp or heads % tp or context % cp or experts % ep:
        raise ValueError("this reference requires divisible layer/head/context/expert axes")
    coordinates = list(product(*(range(axis) for axis in axes)))
    groups = {}
    for index, name in enumerate(("dp", "tp", "pp", "cp", "ep")):
        buckets = {}
        for rank, coordinate in enumerate(coordinates):
            key = coordinate[:index] + coordinate[index + 1:]
            buckets.setdefault(key, []).append(rank)
        groups[name] = list(buckets.values())
    return {"world_size": prod(axes), "coordinates": coordinates, "groups": groups}


def joint_reference(inputs, query_weight, key_weight, value_weight, out_weight,
                    gate_weight, up_weight, down_weight, *, dp=2, tp=2, cp=2):
    """One-process dense DP/TP/CP ownership plus a two-stage attention/FFN boundary.

    DP splits examples; CP splits query positions; TP splits heads and FFN hidden
    channels. PyTorch connects the stage boundary and sums shared gradients.
    No process group, expert routing, or communication overlap is implemented.
    """
    import torch
    batches = inputs.shape[0]
    if gate_weight.shape[1] % tp:
        raise ValueError("FFN width must divide TP degree")
    outputs = []
    for batch_span in balanced_ranges(batches, dp, require_nonempty=True):
        for index in batch_span:
            x = inputs[index]
            query = torch.einsum("td,dhk->thk", x, query_weight)
            key = torch.einsum("td,dhk->thk", x, key_weight)
            value = torch.einsum("td,dhk->thk", x, value_weight)
            stage_boundary = head_parallel_attention(query, key, value, out_weight,
                                                     tp, context_ranks=cp)
            outputs.append(expert_tensor_parallel(stage_boundary, gate_weight, up_weight,
                                                   down_weight, [gate_weight.shape[1] // tp] * tp))
    return torch.stack(outputs)
