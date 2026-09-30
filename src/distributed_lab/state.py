"""State ownership and parameter materialization as CPU reference contracts."""

from .partitions import balanced_ranges


def zero_ledger(parameters: int, ranks: int) -> tuple[float, ...]:
    if parameters < 1 or ranks < 1:
        raise ValueError("positive parameter count and ranks required")
    # BF16 weights, FP32 gradients/master weights/Adam moments: 2+4+4+8.
    return (18 * parameters, (6 + 12 / ranks) * parameters,
            (2 + 16 / ranks) * parameters, 18 * parameters / ranks)


def adam_step(parameters, gradients, moments, variances, step, *, lr=0.01):
    import torch
    if step < 1 or parameters.shape != gradients.shape:
        raise ValueError("positive step and matching parameter/gradient shapes required")
    first = 0.9 * moments + 0.1 * gradients
    second = 0.999 * variances + 0.001 * gradients.square()
    updated = parameters - lr * (first / (1 - 0.9 ** step)) / (
        torch.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
    return updated, first, second


def sharded_adam_step(parameters, gradients, moments, variances, step, ranks):
    """Each owner updates its Adam slice; concatenation is the all-gather contract."""
    import torch
    if parameters.ndim != 1:
        raise ValueError("flat parameter vector required")
    pieces = [adam_step(parameters[list(span)], gradients[list(span)],
                        moments[list(span)], variances[list(span)], step)
              for span in balanced_ranges(len(parameters), ranks, require_nonempty=True)]
    return tuple(torch.cat([piece[index] for piece in pieces]) for index in range(3))


def fsdp_reference(inputs, weight_shards, targets):
    """Gather one linear module, evaluate global mean, return owned gradient slices.

    Shards split the output-feature dimension of a [in,out] matrix. This function
    executes in one CPU process; no FSDP API or concurrent storage claim is made.
    """
    import torch
    complete = torch.cat(weight_shards, dim=1)
    output = inputs @ complete
    loss = (output - targets).square().mean()
    gradients = torch.autograd.grad(loss, weight_shards, retain_graph=True)
    return output, gradients
