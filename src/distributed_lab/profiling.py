"""A small measured baseline; CUDA results require an actual CUDA device."""

import argparse
import json
import platform
from statistics import median
from time import perf_counter


def audit_precision(*, device="cuda", precision="bf16"):
    import torch
    from torch.nn.functional import scaled_dot_product_attention
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for the AMP audit")
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU smoke audit uses FP32")
    torch.manual_seed(53)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    tensors = tuple(torch.randn(1, 2, 8, 16, device=device, requires_grad=True) for _ in range(3))

    def compute(enabled):
        with torch.autocast(device_type=device, dtype=dtype, enabled=enabled):
            output = scaled_dot_product_attention(*tensors, is_causal=True)
            loss = output.float().square().mean()
        gradients = torch.autograd.grad(loss, tensors, retain_graph=True)
        return output.detach(), gradients

    reference, reference_gradients = compute(False)
    output, gradients = compute(precision != "fp32")
    values = torch.tensor([10000., 1., -10000.], device=device)
    reduction_reference = values.sum()
    low_precision_reduction = values.to(dtype).sum()
    return {"device": device, "precision": precision, "output_dtype": str(output.dtype),
            "gradient_dtypes": [str(value.dtype) for value in gradients],
            "output_max_absolute_error": float((output.float() - reference).abs().max()),
            "gradient_max_absolute_error": max(float((left - right).abs().max())
                                                for left, right in zip(gradients, reference_gradients)),
            "output_finite": bool(torch.isfinite(output).all()),
            "gradients_finite": all(bool(torch.isfinite(value).all()) for value in gradients),
            "reduction_reference": float(reduction_reference),
            "reduction_in_selected_dtype": float(low_precision_reduction)}


def run(*, device="cuda", precision="fp32", batch=2, tokens=128, width=256,
        layers=4, segments=1, steps=10, trace=None):
    import torch
    from torch import nn
    from torch.utils.checkpoint import checkpoint_sequential
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device required: CPU timings do not measure GPU throughput")
    if min(batch, tokens, width, layers, segments, steps) < 1 or segments > layers:
        raise ValueError("positive shapes/steps and segments <= layers required")
    if precision not in ("fp32", "bf16", "fp16") or device not in ("cpu", "cuda"):
        raise ValueError("supported device and precision required")
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU reference timing uses FP32; AMP audit requires CUDA")
    torch.manual_seed(47)
    model = nn.Sequential(*(nn.Sequential(nn.Linear(width, width * 4), nn.GELU(),
                                         nn.Linear(width * 4, width)) for _ in range(layers))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randn(batch, tokens, width, device=device)
    targets = torch.randn_like(inputs)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda" and precision == "fp16")

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=dtype, enabled=precision != "fp32"):
            output = (model(inputs) if segments == 1 else
                      checkpoint_sequential(model, segments, inputs, use_reentrant=False))
            loss = (output.float() - targets).square().mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite loss")
        return loss.detach()

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(3):
        step()
    synchronize()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    durations = []
    for _ in range(steps):
        synchronize()
        start = perf_counter()
        loss = step()
        synchronize()
        durations.append(perf_counter() - start)
    report = {"device": torch.cuda.get_device_name() if device == "cuda" else platform.processor() or "CPU",
              "backend": device, "torch": torch.__version__, "cuda": torch.version.cuda,
              "precision": precision, "shape": [batch, tokens, width], "layers": layers,
              "segments": segments, "raw_step_seconds": durations,
              "median_step_seconds": median(durations), "tokens_per_second": batch * tokens / median(durations),
              "last_loss": float(loss), "seed": 47,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else None}
    if trace:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, record_shapes=True,
                                    profile_memory=True) as profiler:
            step()
            synchronize()
        profiler.export_chrome_trace(trace)
        report["trace"] = trace
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    for name, default in (("batch", 2), ("tokens", 128), ("width", 256), ("layers", 4), ("segments", 1), ("steps", 10)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--trace")
    parser.add_argument("--audit", action="store_true")
    arguments = vars(parser.parse_args())
    audit = arguments.pop("audit")
    report = (audit_precision(device=arguments["device"], precision=arguments["precision"])
              if audit else run(**arguments))
    print(json.dumps(report, sort_keys=True))
