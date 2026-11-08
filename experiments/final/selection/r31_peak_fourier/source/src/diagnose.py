import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.data import ImageBank
from src.metrics import pearson
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase
from src.protocol import LEGACY_PROTOCOL, PROTOCOLS, prepare_target, prepare_reconstruction
from src.train import png_roundtrip


def pairwise_pearson(left, right):
    """Return [B,4,4] correlations and validity; row=left, column=right."""
    scores, flags = [], []
    for plane in range(left.shape[1]):
        score, valid = pearson(left[:, plane:plane + 1].expand_as(right), right, return_valid=True)
        scores.append(score)
        flags.append(valid)
    return torch.stack(scores, dim=1), torch.stack(flags, dim=1)


def summarize_matrix(scores, valid):
    mean = np.full(scores.shape[1:], np.nan, dtype=np.float64)
    for row in range(mean.shape[0]):
        for col in range(mean.shape[1]):
            mask = valid[:, row, col]
            if mask.any():
                mean[row, col] = scores[mask, row, col].mean()
    diagonal = np.eye(mean.shape[0], dtype=bool)
    diagonal_values = scores[:, diagonal][valid[:, diagonal]]
    off_diagonal_values = scores[:, ~diagonal][valid[:, ~diagonal]]
    diagonal_mean = float(diagonal_values.mean()) if diagonal_values.size else None
    off_diagonal_mean = float(off_diagonal_values.mean()) if off_diagonal_values.size else None
    return {
        "mean_over_valid": [[None if np.isnan(value) else float(value) for value in row] for row in mean],
        "valid_counts": valid.sum(axis=0).tolist(),
        "invalid_counts": (~valid).sum(axis=0).tolist(),
        "diagonal_mean_over_valid": diagonal_mean,
        "off_diagonal_mean_over_valid": off_diagonal_mean,
        "diagonal_minus_off_diagonal": diagonal_mean - off_diagonal_mean if diagonal_mean is not None and off_diagonal_mean is not None else None,
    }


def summarize_group(arrays):
    result = {"tuples": int(len(arrays["raw_mse"]))}
    for name in ("reconstruction_target", "target_target", "reconstruction_reconstruction"):
        result[name] = summarize_matrix(arrays[name], arrays[name + "_valid"])
    for name in ("raw_mse", "rms_lower_bound", "mse_above_rms_bound", "target_rms", "reconstruction_raw_rms", "metric_target_rms", "png_mse"):
        result[name] = {"mean": float(arrays[name].mean()), "per_plane": arrays[name].mean(axis=0).tolist()}
    return result


def saved_quantization_gaps(run):
    path = run / "metrics.json"
    if not path.exists():
        return {"available": False}
    metrics = json.loads(path.read_text(encoding="utf-8"))
    result = {"available": True, "source": "metrics.json", "protocol": metrics.get("protocol", LEGACY_PROTOCOL), "scope": "original full saved evaluation, not necessarily the diagnostic prefix or override protocol"}
    for name, values in {"aggregate": metrics["aggregate"], **metrics.get("groups", {})}.items():
        floating = values.get("float_ssim")
        phase_quantized = values.get("phase_quantized_ssim")
        png = values.get("ssim")
        result[name] = {
            "float_ssim": floating,
            "phase_quantized_ssim": phase_quantized,
            "png_ssim": png,
            "phase_quantization_ssim_drop": floating - phase_quantized if floating is not None and phase_quantized is not None else None,
            "amplitude_quantization_ssim_drop": phase_quantized - png if phase_quantized is not None and png is not None else None,
        }
    return result


@torch.no_grad()
def diagnose(run, per_group=64, device="cuda", evaluation_protocol=None, output=None):
    run = Path(run)
    if per_group < 1:
        raise ValueError("per_group must be positive")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    input_protocol = config.get("protocol", LEGACY_PROTOCOL)
    protocol = evaluation_protocol or input_protocol
    manifest_path = run / "validation_tuples.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(config["model"], config["width"], config.get("phase_head", "tanh"), position=config.get("position", "none")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    bank = ImageBank(seed=config.get("data_seed", 42))
    propagator = FresnelPropagator().to(device)
    scales = torch.tensor(config["amplitude_scales"], dtype=torch.float32, device=device).reshape(1, 4, 1, 1)
    expected_rms = propagator.field_rms.float()[None] * scales.reshape(1, 4)
    batch_size = config.get("batch_size", 16)
    group_arrays = {}
    groups = {}
    for group, source_records in manifest.items():
        if len(source_records) < per_group:
            raise ValueError(f"Only {len(source_records)} saved validation tuples in {group}; requested {per_group}")
        records = np.asarray(source_records[:per_group], dtype=np.int64)
        flat = records.reshape(-1, 2)
        for dataset in range(len(bank.names)):
            if not np.isin(flat[flat[:, 0] == dataset, 1], bank.val[dataset]).all():
                raise ValueError(f"Saved tuples contain non-validation images in {group}")
        batches = {}
        for start in range(0, len(records), batch_size):
            original_target = bank.fetch(records[start:start + batch_size], device)
            target = prepare_target(original_target, protocol)
            phase = model(prepare_target(original_target, input_protocol))
            phase_codes = png_roundtrip(phase_to_uint8(phase))
            raw = propagator.amplitude(uint8_to_phase(phase_codes)) * scales
            codes = png_roundtrip((prepare_reconstruction(raw, protocol) * 255).round().to(torch.uint8))
            reconstruction = codes.float() / 255
            values = {}
            for name, left, right in (
                ("reconstruction_target", reconstruction, target),
                ("target_target", target, target),
                ("reconstruction_reconstruction", reconstruction, reconstruction),
            ):
                values[name], values[name + "_valid"] = pairwise_pearson(left, right)
            target_rms = original_target.square().mean(dim=(-2, -1)).sqrt()
            raw_mse = (raw - original_target).square().mean(dim=(-2, -1))
            lower_bound = (expected_rms - target_rms).square()
            values.update({
                "raw_mse": raw_mse,
                "rms_lower_bound": lower_bound,
                "mse_above_rms_bound": raw_mse - lower_bound,
                "target_rms": target_rms,
                "metric_target_rms": target.square().mean(dim=(-2, -1)).sqrt(),
                "png_mse": (reconstruction - target).square().mean(dim=(-2, -1)),
                "reconstruction_raw_rms": raw.square().mean(dim=(-2, -1)).sqrt(),
            })
            for key, value in values.items():
                batches.setdefault(key, []).append(value.cpu().numpy())
        group_arrays[group] = {key: np.concatenate(value) for key, value in batches.items()}
        groups[group] = summarize_group(group_arrays[group])
    pooled = {key: np.concatenate([values[key] for values in group_arrays.values()]) for key in next(iter(group_arrays.values()))}
    result = {
        "run": run.name,
        "checkpoint": "best.pt",
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "per_group": per_group,
        "device": device,
        "protocol": protocol,
        "model_input_protocol": input_protocol,
        "matrix_convention": "row=first named quantity/plane; column=second quantity/plane; compare array indices on each plane's own sampling grid",
        "correlation_aggregation": "mean over valid entries with counts explicitly reported",
        "bound_interpretation": "Necessary MSE bound for the scaled raw optical field against original input PNG/255 before peak normalization. Not a physical optimum or proof of target feasibility. Peak-normalized, clipped and PNG MSE are not subject to this raw-field bound; raw_mse is not the v2 metric/loss.",
        "aggregate": summarize_group(pooled),
        "groups": groups,
        "quantization_gaps": saved_quantization_gaps(run),
    }
    destination = Path(output) if output is not None else run / ("diagnostics.json" if evaluation_protocol is None else f"diagnostics_{protocol}.json")
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if previous != result:
            raise FileExistsError(f"Different diagnostics already exist: {destination}")
    else:
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol", choices=PROTOCOLS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    result = diagnose(args.run, args.per_group, args.device, args.protocol, args.output)
    print(json.dumps({"run": result["run"], "checkpoint_step": result["checkpoint_step"], "aggregate": result["aggregate"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
