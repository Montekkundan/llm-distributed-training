"""CPU-only distributed-training partition and numerical contract lab."""

from .context import context_parallel_causal_attention, serial_causal_attention
from .data import data_parallel_linear_gradient, serial_linear_gradient
from .pipeline import serial_chain_loss_gradient, simulate_gpipe
from .tensor import column_parallel_matvec, row_parallel_matvec, serial_matvec

__all__ = [
    "column_parallel_matvec",
    "context_parallel_causal_attention",
    "data_parallel_linear_gradient",
    "row_parallel_matvec",
    "serial_causal_attention",
    "serial_chain_loss_gradient",
    "serial_linear_gradient",
    "serial_matvec",
    "simulate_gpipe",
]
