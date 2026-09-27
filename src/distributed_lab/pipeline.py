"""Fill-drain pipeline contract for a scalar chain, without device execution."""

from __future__ import annotations

from dataclasses import dataclass

from .partitions import balanced_ranges


@dataclass(frozen=True)
class Event:
    phase: str
    microbatch: int
    stage: int


@dataclass(frozen=True)
class PipelineResult:
    stage_layers: tuple[tuple[int, ...], ...]
    events: tuple[Event, ...]
    mean_loss: float
    mean_gradients: tuple[float, ...]


def _validate(weights: list[float], inputs: list[float], targets: list[float]) -> None:
    if not weights or not inputs or len(inputs) != len(targets):
        raise ValueError("Need nonempty layers and paired nonempty inputs and targets")


def serial_chain_loss_gradient(
    weights: list[float], inputs: list[float], targets: list[float],
) -> tuple[float, tuple[float, ...]]:
    """Reference gradient for a chain of scalar multiplications and half-MSE."""
    _validate(weights, inputs, targets)
    gradients = [0.0] * len(weights)
    loss_sum = 0.0
    for x, target in zip(inputs, targets):
        activations = [x]
        for weight in weights:
            activations.append(activations[-1] * weight)
        error = activations[-1] - target
        loss_sum += 0.5 * error * error
        delta = error
        for layer in range(len(weights) - 1, -1, -1):
            gradients[layer] += delta * activations[layer]
            delta *= weights[layer]
    count = len(inputs)
    return loss_sum / count, tuple(gradient / count for gradient in gradients)


def simulate_gpipe(
    weights: list[float], inputs: list[float], targets: list[float], stages: int,
) -> PipelineResult:
    """Execute all forward microbatches, then all backward microbatches.

    The event log expresses stage dependencies; it does not estimate overlap,
    utilization, communication, or GPU runtime.
    """
    _validate(weights, inputs, targets)
    layer_ranges = balanced_ranges(len(weights), stages, require_nonempty=True)
    activations = [[x] for x in inputs]
    events: list[Event] = []

    for microbatch in range(len(inputs)):
        for stage, layers in enumerate(layer_ranges):
            for layer in layers:
                activations[microbatch].append(activations[microbatch][-1] * weights[layer])
            events.append(Event("forward", microbatch, stage))

    gradients = [0.0] * len(weights)
    loss_sum = 0.0
    for microbatch in range(len(inputs) - 1, -1, -1):
        error = activations[microbatch][-1] - targets[microbatch]
        loss_sum += 0.5 * error * error
        delta = error
        for stage in range(len(layer_ranges) - 1, -1, -1):
            for layer in reversed(layer_ranges[stage]):
                gradients[layer] += delta * activations[microbatch][layer]
                delta *= weights[layer]
            events.append(Event("backward", microbatch, stage))

    count = len(inputs)
    return PipelineResult(
        stage_layers=tuple(tuple(layers) for layers in layer_ranges),
        events=tuple(events),
        mean_loss=loss_sum / count,
        mean_gradients=tuple(gradient / count for gradient in gradients),
    )
