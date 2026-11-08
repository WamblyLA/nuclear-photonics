import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.data import ImageBank
from src.diagnose import pairwise_pearson, summarize_matrix
from src.metrics import ssim
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase
from src.protocol import LEGACY_PROTOCOL, PROTOCOLS, prepare_target, prepare_reconstruction
from src.train import png_roundtrip, write_json


def masked_mean(values, mask):
    count = mask.sum((-2, -1))
    return (values * mask).sum((-2, -1)) / count.clamp_min(1), count


def summarize(arrays):
    result = {}
    for key, value in arrays.items():
        if key.endswith("_valid"):
            continue
        if key.endswith("_matrix"):
            result[key] = summarize_matrix(value, arrays[key + "_valid"])
        else:
            result[key] = {"mean": float(value.mean()), "per_plane": value.mean(0).tolist()}
    return result


@torch.no_grad()
def analyze(run, bank, per_group, evaluation_protocol=None):
    saved = json.loads((run / "config.json").read_text(encoding="utf-8"))
    input_protocol = saved.get("protocol", LEGACY_PROTOCOL)
    protocol = evaluation_protocol or input_protocol
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(saved["model"], saved["width"], saved["phase_head"], saved["position"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    propagator = FresnelPropagator()
    scales = torch.tensor(saved["amplitude_scales"]).reshape(1, 4, 1, 1)
    expected_rms = propagator.field_rms.float()[None] * scales.reshape(1, 4)
    manifest = json.loads((run / "validation_tuples.json").read_text(encoding="utf-8"))
    yy, xx = torch.meshgrid(torch.arange(128) - 64, torch.arange(128) - 64, indexing="ij")
    radius = (xx.square() + yy.square()).sqrt()
    spatial_masks = {"center": radius < 32, "outer": radius >= 48}
    groups = {}
    for name, source in manifest.items():
        records = np.asarray(source[:per_group])
        if len(records) != per_group:
            raise ValueError("Not enough saved validation tuples")
        for dataset in range(3):
            flat = records.reshape(-1, 2)
            assert np.isin(flat[flat[:, 0] == dataset, 1], bank.val[dataset]).all()
        batches = {}
        for start in range(0, per_group, 8):
            original_target = bank.fetch(records[start:start + 8], "cpu")
            target = prepare_target(original_target, protocol)
            phase = model(prepare_target(original_target, input_protocol))
            phase_codes = png_roundtrip(phase_to_uint8(phase))
            raw = propagator.amplitude(uint8_to_phase(phase_codes)) * scales
            displayed = prepare_reconstruction(raw, protocol)
            reconstruction = png_roundtrip((displayed * 255).round().to(torch.uint8)).float() / 255
            target_rms = original_target.square().mean((-2, -1)).sqrt()
            raw_rms = raw.square().mean((-2, -1)).sqrt()
            assert torch.allclose(raw_rms, expected_rms.expand_as(raw_rms), atol=1e-6, rtol=1e-6)
            mse = (raw - original_target).square().mean((-2, -1))
            bound = (expected_rms - target_rms).square()
            assert (mse >= bound - 1e-6).all()
            values = {
                "target_rms": target_rms,
                "metric_target_rms": target.square().mean((-2, -1)).sqrt(),
                "raw_rms": raw_rms,
                "png_rms": reconstruction.square().mean((-2, -1)).sqrt(),
                "raw_mse": mse,
                "rms_mse_bound": bound,
                "raw_mse_above_bound": mse - bound,
                "local_ssim": ssim(reconstruction, target),
                "png_mse": (reconstruction - target).square().mean((-2, -1)),
                "clipped_fraction": ((raw if protocol == LEGACY_PROTOCOL else displayed) > 1).float().mean((-2, -1)),
                "raw_above_one_fraction": (raw > 1).float().mean((-2, -1)),
                "phase_near_endpoint_fraction": (phase.abs() > 0.95 * math.pi).float().mean((-2, -1)),
                "phase_mean": phase.mean((-2, -1)),
                "phase_std": phase.std((-2, -1)),
                "phase_min": phase.amin((-2, -1)),
                "phase_max": phase.amax((-2, -1)),
            }
            for label, mask in (("background", target < 0.05), ("foreground", target > 0.1)):
                for kind, image in (("target", target), ("reconstruction", reconstruction)):
                    values[label + "_" + kind + "_mean"], counts = masked_mean(image, mask)
                values[label + "_fraction"] = counts.float() / (128 * 128)
                values[label + "_empty_fraction"] = (counts == 0).float()
                values[label + "_raw_energy_fraction"] = (raw.square() * mask).sum((-2, -1)) / raw.square().sum((-2, -1))
            for label, mask in spatial_masks.items():
                for kind, image in (("target", target), ("reconstruction", reconstruction)):
                    values[label + "_" + kind + "_mean"] = image[..., mask].mean(-1)
                values[label + "_mse"] = (reconstruction[..., mask] - target[..., mask]).square().mean(-1)
            for label, left, right in (("reconstruction_target", reconstruction, target), ("reconstruction_reconstruction", reconstruction, reconstruction), ("target_target", target, target)):
                values[label + "_matrix"], values[label + "_matrix_valid"] = pairwise_pearson(left, right)
            for key, value in values.items():
                batches.setdefault(key, []).append(value.numpy())
        groups[name] = summarize({key: np.concatenate(value) for key, value in batches.items()})
    return {
        "run": run.name,
        "protocol": protocol,
        "model_input_protocol": input_protocol,
        "raw_diagnostics_reference": "raw_mse, target_rms and RMS bound compare the scaled raw optical field with original input PNG/255, before per-image peak normalization; they are not the v2 metric or loss",
        "selected_step": checkpoint["step"],
        "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(),
        "validation_manifest_sha256": hashlib.sha256((run / "validation_tuples.json").read_bytes()).hexdigest(),
        "groups": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--per-group", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("experiments/cycle2_error_analysis.json"))
    parser.add_argument("--protocol", choices=PROTOCOLS)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    configs = [json.loads((run / "config.json").read_text(encoding="utf-8")) for run in args.runs]
    data_seeds = {config.get("data_seed", 42) for config in configs}
    if len(data_seeds) != 1:
        raise ValueError("All runs must use the same data_seed for shared diagnostic image pools")
    data_seed = data_seeds.pop()
    bank = ImageBank(seed=data_seed)
    result = {
        "device": "cpu", "threads": 4, "torch": torch.__version__, "per_group": args.per_group,
        "data_seed": data_seed,
        "scope": "Prefix of saved selection validation, no test images; PNG roundtrips retained",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Means of per-image measurements. Background target<0.05, foreground target>0.1; empty masks contribute zero and their fractions are reported. Phase has one plane. Center r<32, outer r>=48 in output array pixel units. RMS bound is necessary for raw fields, not a feasible optical optimum or PNG bound.",
        "runs": [analyze(run, bank, args.per_group, args.protocol) for run in args.runs],
    }
    write_json(args.output, result)
    for run in result["runs"]:
        for group, values in run["groups"].items():
            print(json.dumps({"run": run["run"], "group": group, **{key: values[key]["mean"] for key in ("target_rms", "raw_mse", "rms_mse_bound", "background_reconstruction_mean", "background_raw_energy_fraction", "local_ssim", "phase_near_endpoint_fraction")}, "correlation_margin": values["reconstruction_target_matrix"]["diagonal_minus_off_diagonal"]}), flush=True)


if __name__ == "__main__":
    main()
