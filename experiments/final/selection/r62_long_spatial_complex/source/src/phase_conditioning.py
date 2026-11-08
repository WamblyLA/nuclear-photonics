import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data import ImageBank
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator
from src.protocol import LEGACY_PROTOCOL, prepare_target
from src.train import write_json


def tanh_conditioning(phase, propagator):
    gradient = 1 - (phase / math.pi).square()
    shifts = []
    for shift in torch.arange(16) * 2 * math.pi / 16:
        shifted = torch.atan2(torch.sin(phase + shift), torch.cos(phase + shift))
        relative = 1 - (shifted / math.pi).square()
        shifts.append({"radians": float(shift), "relative_gradient_mean": float(relative.mean()), "relative_gradient_below_0_1_fraction": float((relative < 0.1).float().mean())})
    best = min(shifts, key=lambda value: value["relative_gradient_below_0_1_fraction"])
    shifted = torch.atan2(torch.sin(phase[:1] + best["radians"]), torch.cos(phase[:1] + best["radians"]))
    return {
        "relative_gradient_mean": float(gradient.mean()),
        "relative_gradient_median": float(gradient.median()),
        "relative_gradient_below_0_1_fraction": float((gradient < 0.1).float().mean()),
        "relative_gradient_below_0_01_fraction": float((gradient < 0.01).float().mean()),
        "best_global_gauge_from_16": best,
        "float_gauge_amplitude_max_diff_first_quad": float((propagator.amplitude(phase[:1]) - propagator.amplitude(shifted)).abs().max()),
        "gauge_grid": shifts,
    }


def circular_conditioning(logits):
    real, imag = logits.chunk(2, dim=1)
    radius = (real.square() + imag.square()).sqrt().flatten()
    gradient = torch.where(radius < 1e-4, torch.zeros_like(radius), radius.clamp_min(1e-4).reciprocal())
    return {
        "radius_min": float(radius.min()), "radius_p001": float(torch.quantile(radius, 0.001)),
        "radius_median": float(radius.median()), "guard_fraction": float((radius < 1e-4).float().mean()),
        "local_gradient_norm_p99": float(torch.quantile(gradient, 0.99)),
        "local_gradient_norm_max": float(gradient.max()),
    }


def collect_phase_logits(model, outputs):
    for name in ("output", "spectral_output", "full_output", "subpixel_output", "cross_phase_output"):
        layer = getattr(model, name, None)
        if layer is not None:
            def collect(module, inputs, value):
                if name == "cross_phase_output":
                    value = value.transpose(1, 2).reshape(value.shape[0], -1, 32, 32)
                    value = F.pixel_shuffle(value, 4)
                outputs.append(value)
            return layer.register_forward_hook(collect)
    raise ValueError("No supported final phase-logit layer found in this model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    saved = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    input_protocol = saved.get("protocol", LEGACY_PROTOCOL)
    model = build_model(saved["model"], saved["width"], saved["phase_head"], saved["position"])
    model.load_state_dict(torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)["model"])
    model.eval()
    outputs = []
    handle = collect_phase_logits(model, outputs) if saved["phase_head"] == "circular" else None
    bank = ImageBank(seed=saved.get("data_seed", 42))
    manifest = json.loads((args.run / "validation_tuples.json").read_text(encoding="utf-8"))
    result = {"run": args.run.name, "device": "cpu", "per_group": args.per_group, "phase_head": saved["phase_head"], "protocol": input_protocol, "model_input_protocol": input_protocol, "groups": {}}
    with torch.no_grad():
        for group, records in manifest.items():
            if len(records) < args.per_group:
                raise ValueError("Not enough saved tuples")
            outputs.clear()
            phases = []
            for start in range(0, args.per_group, 8):
                target = bank.fetch(np.asarray(records[start:min(start + 8, args.per_group)]), "cpu")
                phases.append(model(prepare_target(target, input_protocol)))
            if saved["phase_head"] == "tanh":
                values = tanh_conditioning(torch.cat(phases), FresnelPropagator())
            elif saved["phase_head"] == "circular":
                values = circular_conditioning(torch.cat(outputs))
            else:
                values = {"local_gradient_abs": math.pi, "phase_near_endpoint_fraction": float((torch.cat(phases).abs() > 0.95 * math.pi).float().mean()), "interpretation": "Wrapped scalar head has constant derivative pi; endpoint occupancy is not saturation."}
            result["groups"][group] = values
    if handle is not None:
        handle.remove()
    logits = torch.linspace(-10, 10, 10001, requires_grad=True)
    torch.atan2(torch.sin(math.pi * logits), torch.cos(math.pi * logits)).sum().backward()
    result["wrapped_head_autograd_check"] = {"sample_count": logits.numel(), "logits_interval": [-10, 10], "maximum_abs_error_from_pi": float((logits.grad - math.pi).abs().max()), "all_gradients_finite": bool(torch.isfinite(logits.grad).all())}
    result["interpretation"] = "Tanh relative derivative divides by pi; circular local derivative norm is 1/r outside origin guard. Gauge shifts preserve float optical amplitude and only diagnose conditioning, not quality improvement. No optimization performed."
    write_json(args.output, result)
    print(json.dumps({"run": args.run.name, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
