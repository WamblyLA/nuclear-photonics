"""Neural-only single-plane diagnostic, never a four-plane submission model.

Only one input image and one physical reconstruction enter the objective.
All other input channels are zero. Training updates one shared neural network
on real TRAIN images, with no input-specific mask parameters or refinement.
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.amplitude_losses import balanced_pixel_mse
from src.data import ImageBank
from src.metrics import pearson, ssim
from src.models.transport import TransportPhaseGenerator
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase
from src.protocol import PEAK_PROTOCOL, prepare_reconstruction, prepare_target
from src.shape_losses import excess_concentration_loss
from src.train import image_metrics, png_roundtrip, snapshot_source, write_json


@dataclass
class Config:
    run: str = "single_plane_probe"
    active_plane: int = 0
    width: int = 16
    steps: int = 5000
    batch_size: int = 16
    lr: float = 2e-4
    seed: int = 42
    data_seed: int = 42
    domain: str = "mixed"
    val_per_group: int = 64
    eval_every: int = 1000
    device: str = "cuda"
    ssim_weight: float = 1.0
    pearson_weight: float = 0.5
    balanced_weight: float = 2.0
    concentration_weight: float = 0.1
    model: str = "transport_complex"
    phase_head: str = "circular"
    position: str = "xy"
    protocol: str = PEAK_PROTOCOL
    diagnostic_only: bool = True


def active_inputs(target, active_plane):
    inputs = torch.zeros_like(target)
    inputs[:, active_plane:active_plane + 1] = target[:, active_plane:active_plane + 1]
    return inputs


def objective_terms(raw, reconstructed, target, cfg):
    mse = (reconstructed - target).square().mean((-2, -1))
    structural = ssim(reconstructed, target)
    correlation = pearson(reconstructed, target)
    balanced = balanced_pixel_mse(reconstructed, target)
    concentration = excess_concentration_loss(raw, target)
    objective = mse + cfg.ssim_weight * (1 - structural) + cfg.pearson_weight * (1 - correlation)
    objective = objective + cfg.balanced_weight * balanced + cfg.concentration_weight * concentration
    return {"objective": objective, "mse": mse, "ssim": structural, "pearson": correlation,
            "balanced_mse": balanced, "excess_concentration": concentration}


def summarize_active(values):
    result = {"invalid_pearson_count": int((~values["pearson_valid"]).sum())}
    for key, array in values.items():
        if key == "pearson_valid":
            continue
        result[key] = None if key == "pearson" and not values["pearson_valid"].all() else float(array.mean())
        result[key + "_p10"] = None if result[key] is None else float(np.quantile(array, 0.1))
    return result


def save_active_example(directory, target, phase_codes, reconstruction_codes, active_plane):
    directory.mkdir(parents=True, exist_ok=True)
    target_image = Image.fromarray((target[0, 0].cpu().numpy() * 255).round().astype(np.uint8))
    phase_image = Image.fromarray(phase_codes[0, 0].cpu().numpy())
    reconstructed_image = Image.fromarray(reconstruction_codes[0, 0].cpu().numpy())
    target_image.save(directory / f"target_plane_{active_plane + 1}.png")
    phase_image.save(directory / "phase.png")
    reconstructed_image.save(directory / f"reconstruction_plane_{active_plane + 1}.png")
    grid = Image.new("L", (3 * 128, 128))
    for column, image in enumerate((target_image, reconstructed_image, phase_image)):
        grid.paste(image, (column * 128, 0))
    grid.save(directory / "comparison.png")


@torch.no_grad()
def evaluate_active(model, bank, physics, manifest, cfg, output=None):
    model.eval()
    arrays = {}
    groups = {}
    plane = slice(cfg.active_plane, cfg.active_plane + 1)
    for group, records in manifest.items():
        batches = {}
        for start in range(0, len(records), cfg.batch_size):
            all_targets = prepare_target(bank.fetch(records[start:start + cfg.batch_size], cfg.device), PEAK_PROTOCOL)
            target = all_targets[:, plane]
            phase = model(active_inputs(all_targets, cfg.active_plane))
            phase_codes = png_roundtrip(phase_to_uint8(phase))
            decoded_phase = uint8_to_phase(phase_codes)
            raw = physics.amplitude(decoded_phase)[:, plane]
            displayed = prepare_reconstruction(raw, PEAK_PROTOCOL)
            reconstruction_codes = png_roundtrip((displayed * 255).round().to(torch.uint8))
            reconstruction = reconstruction_codes.float() / 255
            scores = image_metrics(reconstruction, target)
            scores.update(objective_terms(raw, reconstruction, target, cfg))
            foreground = target > 0.1
            background = ~foreground
            scores["foreground_mean"] = (reconstruction * foreground).sum((-2, -1)) / foreground.sum((-2, -1)).clamp_min(1)
            scores["target_foreground_mean"] = (target * foreground).sum((-2, -1)) / foreground.sum((-2, -1)).clamp_min(1)
            scores["background_mean"] = (reconstruction * background).sum((-2, -1)) / background.sum((-2, -1)).clamp_min(1)
            scores["raw_peak"] = raw.amax((-2, -1))
            float_reconstruction = prepare_reconstruction(physics.amplitude(phase)[:, plane], PEAK_PROTOCOL)
            scores["float_ssim"] = ssim(float_reconstruction, target)
            for key, value in scores.items():
                batches.setdefault(key, []).append(value.cpu().numpy().reshape(-1))
            if output is not None and start == 0:
                save_active_example(output / "examples" / group, target, phase_codes, reconstruction_codes, cfg.active_plane)
        arrays[group] = {key: np.concatenate(value) for key, value in batches.items()}
        groups[group] = summarize_active(arrays[group])
    pooled = {key: np.concatenate([group[key] for group in arrays.values()]) for key in next(iter(arrays.values()))}
    aggregate = summarize_active(pooled)
    result = {
        "diagnostic_only": True, "active_plane": cfg.active_plane, "evaluation_planes": [cfg.active_plane],
        "scope": "One active plane only; not comparable to four-plane challenge scores.",
        "active_plane_aggregate": aggregate, "aggregate": aggregate, "groups": groups,
        "active_images": len(pooled["ssim"]), "protocol": PEAK_PROTOCOL, "test_used": False,
    }
    if output is not None:
        np.savez_compressed(output / "per_example_metrics.npz", **{f"{group}__{key}": value for group, scores in arrays.items() for key, value in scores.items()})
        write_json(output / "metrics.json", result)
    return result


def train(cfg):
    if not 0 <= cfg.active_plane < 4 or min(cfg.width, cfg.steps, cfg.batch_size, cfg.val_per_group, cfg.eval_every) < 1:
        raise ValueError("Active plane must be 0..3 and training sizes must be positive")
    if cfg.lr <= 0 or min(cfg.ssim_weight, cfg.pearson_weight, cfg.balanced_weight, cfg.concentration_weight) < 0:
        raise ValueError("Learning rate must be positive and loss weights nonnegative")
    if not cfg.diagnostic_only or cfg.model != "transport_complex" or cfg.phase_head != "circular" or cfg.position != "xy" or cfg.protocol != PEAK_PROTOCOL:
        raise ValueError("This diagnostic requires its declared transport_complex/v2 configuration")
    output = Path("outputs") / cfg.run
    if output.exists():
        raise FileExistsError(output)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = TransportPhaseGenerator(width=cfg.width, phase_decoder="complex").to(cfg.device)
    physics = FresnelPropagator().to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    bank = ImageBank(seed=cfg.data_seed)
    manifest = bank.validation_manifest(cfg.val_per_group)
    config = {**asdict(cfg), "evaluation_planes": [cfg.active_plane], "selection_metric": "minimum active-plane validation objective",
              "scope": "One active plane only; diagnostic model must never be submitted.",
              "parameters": sum(parameter.numel() for parameter in model.parameters()), "torch": torch.__version__,
              "amplitude_scales": [1.0] * 4, "test_used": False}
    write_json(output / "config.json", config)
    write_json(output / "validation_tuples.json", {key: records.tolist() for key, records in manifest.items()})
    bank.save_manifest(output / "data_manifest.json", cfg.val_per_group)
    snapshot_source(output)
    write_json(output / "initial_metrics.json", evaluate_active(model, bank, physics, manifest, cfg))
    rng = np.random.default_rng(cfg.seed + 1000)
    best_objective = float("inf")
    training_seconds = 0.0
    plane = slice(cfg.active_plane, cfg.active_plane + 1)
    for step in range(1, cfg.steps + 1):
        model.train()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        all_targets = prepare_target(bank.fetch(bank.sample_train(cfg.batch_size, rng, cfg.domain), cfg.device), PEAK_PROTOCOL)
        target = all_targets[:, plane]
        optimizer.zero_grad(set_to_none=True)
        phase = model(active_inputs(all_targets, cfg.active_plane))
        raw = physics.amplitude(phase)[:, plane]
        reconstructed = prepare_reconstruction(raw, PEAK_PROTOCOL)
        terms = objective_terms(raw, reconstructed, target, cfg)
        loss = terms["objective"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite diagnostic loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - tick
        if step == 1 or step % 50 == 0 or step == cfg.steps:
            entry = {"step": step, "diagnostic_only": True, "active_plane": cfg.active_plane,
                     **{key: float(value.detach().mean()) for key, value in terms.items()},
                     "grad_norm": float(gradient), "training_seconds": training_seconds}
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps({"run": cfg.run, **entry}), flush=True)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            validation = evaluate_active(model, bank, physics, manifest, cfg)
            scores = validation["active_plane_aggregate"]
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "validation": validation}) + "\n")
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                          "step": step, "training_seconds": training_seconds, "diagnostic_only": True,
                          "active_plane": cfg.active_plane, "evaluation_planes": [cfg.active_plane]}
            torch.save(checkpoint, output / "last.pt")
            if scores["objective"] < best_objective:
                best_objective = scores["objective"]
                torch.save(checkpoint, output / "best.pt")
            print(json.dumps({"run": cfg.run, "step": step, "diagnostic_only": True,
                              "active_plane": cfg.active_plane, "active_ssim": scores["ssim"],
                              "active_pearson": scores["pearson"], "active_objective": scores["objective"]}), flush=True)
    selected = torch.load(output / "best.pt", map_location=cfg.device, weights_only=False)
    model.load_state_dict(selected["model"])
    result = evaluate_active(model, bank, physics, manifest, cfg, output=output)
    result.update({"selected_step": selected["step"], "selected_checkpoint": "best.pt", "training_seconds": training_seconds,
                   "total_seconds": time.perf_counter() - started, "selection_metric": config["selection_metric"]})
    write_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for key in ("run", "device", "domain"):
        parser.add_argument("--" + key.replace("_", "-"), default=getattr(defaults, key))
    for key in ("active_plane", "width", "steps", "batch_size", "seed", "data_seed", "val_per_group", "eval_every"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=getattr(defaults, key))
    parser.add_argument("--lr", type=float, default=defaults.lr)
    train(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
