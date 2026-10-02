"""Dependency-valid CPU clocks; tick counts are not measured GPU durations."""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Scheduled:
    stage: int
    microbatch: int
    kind: str
    start: int
    end: int


def one_f_one_b(stages: int, microbatches: int, *, forward_ticks=1, backward_ticks=1) -> tuple[Scheduled, ...]:
    if min(stages, microbatches, forward_ticks, backward_ticks) < 1:
        raise ValueError("positive stages, microbatches, and durations required")
    orders = []
    for stage in range(stages):
        warm = min(stages - stage - 1, microbatches)
        order = [("F", j) for j in range(warm)]
        for j in range(microbatches - warm):
            order.extend((("F", warm + j), ("B", j)))
        order.extend(("B", j) for j in range(microbatches - warm, microbatches))
        orders.append(order)
    indices = [{event: index for index, event in enumerate(order)} for order in orders]

    @lru_cache(None)
    def finish(stage, index):
        kind, batch = orders[stage][index]
        dependencies = [finish(stage, index - 1) if index else 0]
        if kind == "F" and stage:
            dependencies.append(finish(stage - 1, indices[stage - 1][("F", batch)]))
        if kind == "B":
            dependencies.append(finish(stage, indices[stage][("F", batch)]))
            if stage < stages - 1:
                dependencies.append(finish(stage + 1, indices[stage + 1][("B", batch)]))
        return max(dependencies) + (forward_ticks if kind == "F" else backward_ticks)

    result = tuple(Scheduled(stage, batch, kind, finish(stage, index) -
                             (forward_ticks if kind == "F" else backward_ticks),
                             finish(stage, index))
                   for stage, order in enumerate(orders)
                   for index, (kind, batch) in enumerate(order))
    validate(result, stages, microbatches, split=False)
    return result


def split_backward(stages: int, microbatches: int) -> tuple[Scheduled, ...]:
    """Prioritize input backward, then forward, then delayed weight backward.

    One F/B/W operation consumes one unit on each stage. This is a legal
    split-backward reference policy, not ZB-H1, ZB-H2, or DualPipe.

    The policy has no activation-memory limit, and that is intentional. A forward
    runs as soon as its upstream forward is done, and W runs only when no B or F is
    ready, so every stage admits all m microbatches before its first W frees one.
    The saved-input peak is therefore m on every stage ([m] * p), where 1F1B has
    min(p - s, m) on stage s. For m >= p the makespan equals the lower bound
    (p - 1) + 3 m for unit F, B, W (see costs.split_backward_makespan_lower_bound),
    so the bubble beyond the start-up delay is zero, but only because memory is
    unbounded. Published Zero Bubble schedules (arXiv 2401.10241) fix an activation
    budget (ZB-H1 near the 1F1B level, ZB-H2 a larger multiple) and trade bubble
    against it. Read the peaks and the makespan here as the unconstrained end of
    that trade, not as ZB-H1 or ZB-H2 results. The model also keeps a whole saved
    input per microbatch until W, which overstates what a real B pass can free.
    """
    if stages < 1 or microbatches < 1:
        raise ValueError("positive stages and microbatches required")
    complete = {}
    pending = {(stage, batch, kind) for stage in range(stages)
               for batch in range(microbatches) for kind in ("F", "B", "W")}
    events = []
    tick = 0
    while pending:
        ready = []
        for stage in range(stages):
            for kind in ("B", "F", "W"):
                options = []
                for batch in range(microbatches):
                    key = (stage, batch, kind)
                    if key not in pending:
                        continue
                    dependencies = []
                    if kind == "F" and stage:
                        dependencies.append((stage - 1, batch, "F"))
                    if kind in ("B", "W"):
                        dependencies.append((stage, batch, "F"))
                    if kind == "B" and stage < stages - 1:
                        dependencies.append((stage + 1, batch, "B"))
                    if kind == "W":
                        dependencies.append((stage, batch, "B"))
                    if all(dependency in complete for dependency in dependencies):
                        options.append(key)
                if options:
                    ready.append(min(options))
                    break
        if not ready:
            raise RuntimeError("schedule dependency deadlock")
        for stage, batch, kind in ready:
            event = Scheduled(stage, batch, kind, tick, tick + 1)
            events.append(event)
        for key in ready:
            complete[key] = tick + 1
            pending.remove(key)
        tick += 1
    result = tuple(events)
    validate(result, stages, microbatches, split=True)
    return result


def validate(events, stages, microbatches, *, split):
    lookup = {(event.stage, event.microbatch, event.kind): event for event in events}
    expected = {(stage, batch, kind) for stage in range(stages)
                for batch in range(microbatches) for kind in (("F", "B", "W") if split else ("F", "B"))}
    if len(lookup) != len(events) or set(lookup) != expected:
        raise ValueError("missing or duplicate schedule operation")
    occupied = set()
    for event in events:
        slots = {(event.stage, tick) for tick in range(event.start, event.end)}
        if event.end <= event.start or slots & occupied:
            raise ValueError("stage resource overlap or invalid duration")
        occupied.update(slots)
        dependencies = []
        if event.kind == "F" and event.stage:
            dependencies.append((event.stage - 1, event.microbatch, "F"))
        if event.kind in ("B", "W"):
            dependencies.append((event.stage, event.microbatch, "F"))
        if event.kind == "B" and event.stage < stages - 1:
            dependencies.append((event.stage + 1, event.microbatch, "B"))
        if event.kind == "W":
            dependencies.append((event.stage, event.microbatch, "B"))
        if any(lookup[key].end > event.start for key in dependencies):
            raise ValueError("operation executes before its dependency")
    return True


def activation_peaks(events, stages, *, split=False):
    """Count saved inputs until complete backward (B), or delayed W."""
    peaks = []
    for stage in range(stages):
        live = peak = 0
        for event in sorted((e for e in events if e.stage == stage), key=lambda e: e.start):
            live += 1 if event.kind == "F" else -1 if event.kind == ("W" if split else "B") else 0
            if live < 0:
                raise ValueError("activation released before creation")
            peak = max(peak, live)
        if live:
            raise ValueError("saved activation leaked at flush")
        peaks.append(peak)
    return peaks
