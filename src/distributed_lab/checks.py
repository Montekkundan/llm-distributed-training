"""Run the four reference-parity checks without third-party packages."""

from __future__ import annotations

import json
from math import isclose

from .context import context_parallel_causal_attention, serial_causal_attention
from .data import data_parallel_linear_gradient, serial_linear_gradient
from .pipeline import serial_chain_loss_gradient, simulate_gpipe
from .tensor import column_parallel_matvec, row_parallel_matvec, serial_matvec


def _same(left: float, right: float) -> bool:
    return isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def run_checks() -> dict[str, bool]:
    samples = [(1.0, 2.0), (2.0, -1.0), (-3.0, 4.0), (0.5, 0.2), (4.0, -2.0)]
    serial_gradient, serial_loss = serial_linear_gradient(samples, 0.7)
    data_result = data_parallel_linear_gradient(samples, 0.7, 3)
    data_ok = _same(data_result.global_mean_gradient, serial_gradient) and _same(
        data_result.global_mean_loss, serial_loss
    )

    weights, inputs, targets = [0.7, -1.2, 0.8, 1.5], [1.0, -2.0, 0.4], [0.5, 1.0, -0.2]
    serial_pipeline_loss, serial_pipeline_gradients = serial_chain_loss_gradient(weights, inputs, targets)
    pipeline_result = simulate_gpipe(weights, inputs, targets, 2)
    pipeline_ok = _same(pipeline_result.mean_loss, serial_pipeline_loss) and all(
        _same(actual, expected)
        for actual, expected in zip(pipeline_result.mean_gradients, serial_pipeline_gradients)
    )

    matrix, vector = [[1.0, -2.0, 0.5], [3.0, 4.0, -1.0], [-2.0, 0.0, 2.0]], [0.2, -1.0, 3.0]
    expected_vector = serial_matvec(matrix, vector)
    column_result = column_parallel_matvec(matrix, vector, 2)
    row_result = row_parallel_matvec(matrix, vector, 2)
    tensor_ok = all(
        _same(actual, expected)
        for output in (column_result.output, row_result.output)
        for actual, expected in zip(output, expected_vector)
    )

    queries = [0.2, -1.0, 3.0, 0.1, -0.4]
    keys = [-0.5, 0.8, 1.2, -1.4, 0.3]
    values = [1.0, 2.0, -3.0, 4.0, 0.5]
    expected_attention = serial_causal_attention(queries, keys, values)
    context_result = context_parallel_causal_attention(queries, keys, values, 3)
    context_ok = all(_same(actual, expected) for actual, expected in zip(context_result.output, expected_attention))

    return {"data": data_ok, "pipeline": pipeline_ok, "tensor": tensor_ok, "context": context_ok}


def main() -> None:
    results = run_checks()
    print(json.dumps(results, sort_keys=True))
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
