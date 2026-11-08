"""CPU audit of contrast and energy concentration on saved selection inputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.amplitude_losses import amplitude_cosine_loss
from src.data import ImageBank
from src.diagnose import pairwise_pearson, summarize_matrix
from src.metrics import pearson, ssim
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase
from src.protocol import LEGACY_PROTOCOL, prepare_target, prepare_reconstruction
from src.train import png_roundtrip, write_json


def masked_mean(image, mask):
    count = mask.sum((-2, -1))
    return (image * mask).sum((-2, -1)) / count.clamp_min(1)


def summarize(arrays):
    summary = {}
    for name, array in arrays.items():
        if name.endswith("_valid"):
            continue
        if name == "reconstruction_target":
            summary[name] = summarize_matrix(array, arrays[name + "_valid"])
            continue
        validity = arrays.get(name + "_valid")
        values = array if validity is None else array[validity]
        summary[name] = {
            "mean": float(values.mean()) if values.size else None,
            "median": float(np.median(values)) if values.size else None,
            "p10": float(np.quantile(values, 0.1)) if values.size else None,
            "p90": float(np.quantile(values, 0.9)) if values.size else None,
            "per_plane": [float(array[:, plane].mean()) if validity is None else (float(array[:, plane][validity[:, plane]].mean()) if validity[:, plane].any() else None) for plane in range(array.shape[1])],
        }
        if validity is not None:
            summary[name]["invalid_count"] = int((~validity).sum())
    return summary


@torch.no_grad()
def diagnose_hotspots(run, bank, manifest, foreground_threshold=0.1, checkpoint_path=None):
    saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
    protocol = saved.get("protocol", LEGACY_PROTOCOL)
    checkpoint_path = checkpoint_path or run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    embedded = checkpoint.get("config", saved)
    for key in ("model", "width", "phase_head", "position", "protocol", "amplitude_scales", "data_seed"):
        if key in embedded and key in saved and embedded[key] != saved[key]:
            raise ValueError(f"Checkpoint/run config mismatch for {key}")
    model = build_model(saved["model"], saved["width"], saved["phase_head"], saved["position"]).eval()
    model.load_state_dict(checkpoint["model"])
    physics = FresnelPropagator()
    scales = torch.tensor(saved["amplitude_scales"]).reshape(1, 4, 1, 1)
    groups, group_arrays = {}, {}
    for name, records in manifest.items():
        batches = {}
        for start in range(0, len(records), 8):
            original = bank.fetch(records[start:start + 8], "cpu")
            target = prepare_target(original, protocol)
            phase = model(prepare_target(original, protocol))
            phase = uint8_to_phase(png_roundtrip(phase_to_uint8(phase)))
            raw = physics.amplitude(phase) * scales
            displayed = prepare_reconstruction(raw, protocol)
            prediction = png_roundtrip((displayed * 255).round().to(torch.uint8)).float() / 255
            foreground = target > foreground_threshold
            background = ~foreground
            energy = raw.square()
            flat_energy = energy.flatten(2)
            top_pixels = max(1, math.ceil(flat_energy.shape[-1] * 0.001))
            peaks = raw.flatten(2).argmax(-1, keepdim=True)
            peak_target = target.flatten(2).gather(-1, peaks).squeeze(-1)
            correlation, valid = pearson(prediction, target, return_valid=True)
            pairwise, pairwise_valid = pairwise_pearson(prediction, target)
            direction_valid = (raw.square().sum((-2, -1)) > 0) & (target.square().sum((-2, -1)) > 0)
            foreground_present = foreground.any(-1).any(-1)
            background_present = background.any(-1).any(-1)
            values = {
                "local_ssim": ssim(prediction, target), "pearson": correlation, "pearson_valid": valid,
                "png_mse": (prediction - target).square().mean((-2, -1)),
                "raw_peak": raw.amax((-2, -1)),
                "amplitude_cosine": 1 - amplitude_cosine_loss(raw, target), "amplitude_cosine_valid": direction_valid,
                "prediction_mean": prediction.mean((-2, -1)), "target_mean": target.mean((-2, -1)),
                "prediction_foreground_mean": masked_mean(prediction, foreground), "prediction_foreground_mean_valid": foreground_present,
                "target_foreground_mean": masked_mean(target, foreground), "target_foreground_mean_valid": foreground_present,
                "prediction_background_mean": masked_mean(prediction, background), "prediction_background_mean_valid": background_present,
                "target_background_mean": masked_mean(target, background), "target_background_mean_valid": background_present,
                "foreground_area_fraction": foreground.float().mean((-2, -1)),
                "raw_energy_foreground_fraction": (energy * foreground).sum((-2, -1)) / energy.sum((-2, -1)).clamp_min(1e-12),
                "peak_in_foreground_fraction": (peak_target > foreground_threshold).float(),
                "target_amplitude_at_prediction_peak": peak_target,
                "top_0_1_percent_raw_energy_fraction": flat_energy.topk(top_pixels, dim=-1).values.sum(-1) / flat_energy.sum(-1).clamp_min(1e-12),
                "reconstruction_target": pairwise, "reconstruction_target_valid": pairwise_valid,
            }
            for key, value in values.items():
                batches.setdefault(key, []).append(value.numpy())
        group_arrays[name] = {key: np.concatenate(values) for key, values in batches.items()}
        groups[name] = summarize(group_arrays[name])
    pooled = {key: np.concatenate([values[key] for values in group_arrays.values()]) for key in next(iter(group_arrays.values()))}
    return {
        "run": run.name, "selected_step": checkpoint["step"], "protocol": protocol,
        "model_input_protocol": protocol, "checkpoint_file": checkpoint_path.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "amplitude_scales": saved["amplitude_scales"], "aggregate": summarize(pooled), "groups": groups,
    }, group_arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", type=Path)
    parser.add_argument("--per-group", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("experiments/night_hotspot_diagnostics.json"))
    args = parser.parse_args()
    if args.per_group < 1:
        raise ValueError("per_group must be positive")
    if args.checkpoints and len(args.checkpoints) != len(args.runs):
        parser.error("--checkpoints requires one checkpoint per run, in the same order")
    if args.output.exists() or args.output.with_suffix(".npz").exists():
        raise FileExistsError(args.output)
    configs = [json.loads((run / "config.json").read_text(encoding="utf-8")) for run in args.runs]
    seeds = {saved.get("data_seed", 42) for saved in configs}
    protocols = {saved.get("protocol", LEGACY_PROTOCOL) for saved in configs}
    if len(seeds) != 1 or len(protocols) != 1:
        raise ValueError("This comparison requires common data split and image protocol")
    if any(saved["amplitude_scales"] != configs[0]["amplitude_scales"] for saved in configs[1:]):
        raise ValueError("Raw peak comparisons require the same saved amplitude scales")
    torch.set_num_threads(4)
    bank = ImageBank(seed=next(iter(seeds)))
    manifests = []
    for run in args.runs:
        saved = json.loads((run / "validation_tuples.json").read_text(encoding="utf-8"))
        manifest = {name: np.asarray(records[:args.per_group], dtype=np.int64) for name, records in saved.items()}
        if any(len(records) != args.per_group for records in manifest.values()):
            raise ValueError("Not enough saved selection tuples")
        flat = np.concatenate(list(manifest.values())).reshape(-1, 2)
        for dataset in range(3):
            if not np.isin(flat[flat[:, 0] == dataset, 1], bank.val[dataset]).all():
                raise ValueError("Diagnostic inputs must be validation sources")
        manifests.append(manifest)
    if any(list(value) != list(manifests[0]) or any(not np.array_equal(value[name], manifests[0][name]) for name in manifests[0]) for value in manifests[1:]):
        raise ValueError("All compared models must use identical diagnostic quadruples")
    prefix = {name: records.tolist() for name, records in manifests[0].items()}
    result = {
        "scope": "Same saved selection prefix, not a fresh holdout; no test images loaded", "test_used": False,
        "device": "cpu", "threads": 4, "torch": torch.__version__, "per_group": args.per_group,
        "quadruples_per_model": args.per_group * len(prefix), "manifest": prefix,
        "manifest_sha256": hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Amplitude metrics after phase/reconstruction PNG roundtrips. Foreground target>0.1; background is its complement. Raw energy means amplitude squared; top0.1% uses ceil(16384*0.001)=17 pixels. Peak location uses the first argmax if tied. Undefined directional/region means carry invalid counts. Raw peaks share the same fixed amplitude scales; displayed images are independently peak-normalized for v2.",
        "runs": [],
    }
    arrays = {}
    checkpoints = args.checkpoints or [run / "best.pt" for run in args.runs]
    for run, checkpoint_path in zip(args.runs, checkpoints):
        metrics, values = diagnose_hotspots(run, bank, manifests[0], checkpoint_path=checkpoint_path)
        result["runs"].append(metrics)
        arrays.update({f"{run.name}__{group}__{key}": array for group, entries in values.items() for key, array in entries.items()})
        keys = ("local_ssim", "pearson", "raw_peak", "amplitude_cosine", "prediction_foreground_mean", "target_foreground_mean", "prediction_background_mean", "raw_energy_foreground_fraction", "peak_in_foreground_fraction", "top_0_1_percent_raw_energy_fraction")
        print(json.dumps({"run": run.name, **{key: metrics["aggregate"][key]["mean"] for key in keys}, "correlation_margin": metrics["aggregate"]["reconstruction_target"]["diagonal_minus_off_diagonal"]}), flush=True)
    np.savez_compressed(args.output.with_suffix(".npz"), **arrays)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
