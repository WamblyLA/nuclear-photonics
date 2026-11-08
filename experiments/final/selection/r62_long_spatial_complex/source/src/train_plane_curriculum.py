"""Random-depth single-plane pretraining of one shared neural network.

Each example activates exactly one input channel and one reconstruction loss.
These diagnostic checkpoints are initialization candidates for ordinary joint
four-plane training, never directly usable as a four-plane submission.
"""

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.data import ImageBank
from src.models.transport import TransportPhaseGenerator
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase
from src.protocol import PEAK_PROTOCOL, prepare_reconstruction, prepare_target
from src.train import image_metrics, png_roundtrip, snapshot_source, write_json
from src.train_single_plane_probe import Config as ProbeConfig
from src.train_single_plane_probe import objective_terms, save_active_example, summarize_active


@dataclass
class Config(ProbeConfig):
    run: str = "random_plane_curriculum"
    active_plane: int | None = None
    random_active_plane: bool = True
    threads: int = 4


def check_active_planes(images, planes):
    if images.ndim != 4 or images.shape[1] != 4:
        raise ValueError("Expected images shaped [batch, 4, height, width]")
    if planes.shape != (images.shape[0],) or planes.dtype != torch.int64:
        raise ValueError("Active planes must be an int64 vector with one index per example")
    if planes.device != images.device or (planes < 0).any() or (planes > 3).any():
        raise ValueError("Active planes must be on the image device and lie in 0..3")


def select_active_planes(images, planes):
    check_active_planes(images, planes)
    index = planes[:, None, None, None].expand(-1, 1, *images.shape[-2:])
    return images.gather(1, index)


def masked_plane_inputs(images, planes):
    check_active_planes(images, planes)
    mask = torch.nn.functional.one_hot(planes, num_classes=4).to(images.dtype)
    return images * mask[..., None, None]


def validation_planes(count, start=0, device="cpu"):
    return torch.arange(start, start + count, dtype=torch.int64, device=device) % 4


def active_loss_terms(all_amplitudes, all_targets, planes, cfg):
    raw = select_active_planes(all_amplitudes, planes)
    target = select_active_planes(all_targets, planes)
    reconstructed = prepare_reconstruction(raw, PEAK_PROTOCOL)
    return objective_terms(raw, reconstructed, target, cfg)


@torch.no_grad()
def evaluate_curriculum(model, bank, physics, manifest, cfg, output=None):
    model.eval()
    arrays, assignments, groups = {}, {}, {}
    for group, records in manifest.items():
        if len(records) % 4:
            raise ValueError("Balanced curriculum validation requires each group size divisible by four")
        batches = {}
        assignments[group] = np.arange(len(records), dtype=np.int64) % 4
        for start in range(0, len(records), cfg.batch_size):
            targets = prepare_target(bank.fetch(records[start:start + cfg.batch_size], cfg.device), PEAK_PROTOCOL)
            planes = validation_planes(len(targets), start, cfg.device)
            target = select_active_planes(targets, planes)
            phase = model(masked_plane_inputs(targets, planes))
            phase_codes = png_roundtrip(phase_to_uint8(phase))
            raw = select_active_planes(physics.amplitude(uint8_to_phase(phase_codes)), planes)
            displayed = prepare_reconstruction(raw, PEAK_PROTOCOL)
            reconstruction_codes = png_roundtrip((displayed * 255).round().to(torch.uint8))
            reconstruction = reconstruction_codes.float() / 255
            scores = image_metrics(reconstruction, target)
            scores.update(objective_terms(raw, reconstruction, target, cfg))
            foreground = target > 0.1
            scores["foreground_mean"] = (reconstruction * foreground).sum((-2, -1)) / foreground.sum((-2, -1)).clamp_min(1)
            scores["target_foreground_mean"] = (target * foreground).sum((-2, -1)) / foreground.sum((-2, -1)).clamp_min(1)
            scores["raw_peak"] = raw.amax((-2, -1))
            for key, value in scores.items():
                batches.setdefault(key, []).append(value.cpu().numpy().reshape(-1))
            if output is not None and start == 0:
                save_active_example(output / "examples" / group, target, phase_codes, reconstruction_codes, int(planes[0]))
        arrays[group] = {key: np.concatenate(value) for key, value in batches.items()}
        groups[group] = summarize_active(arrays[group])
    pooled = {key: np.concatenate([group[key] for group in arrays.values()]) for key in next(iter(arrays.values()))}
    pooled_planes = np.concatenate(list(assignments.values()))
    aggregate = summarize_active(pooled)
    per_plane = {str(plane): {"active_images": int((pooled_planes == plane).sum()),
                            **summarize_active({key: values[pooled_planes == plane] for key, values in pooled.items()})}
                 for plane in range(4)}
    result = {"diagnostic_only": True, "random_active_plane": True, "active_plane": None,
              "evaluation_planes": [0, 1, 2, 3], "joint_four_plane_evaluation": False,
              "scope": "Exactly one active plane per example; never comparable to joint four-plane challenge scores.",
              "active_plane_assignment": "row_index % 4 within each group", "active_plane_aggregate": aggregate,
              "aggregate": aggregate, "groups": groups, "by_active_plane": per_plane,
              "active_images": len(pooled["ssim"]), "protocol": PEAK_PROTOCOL, "test_used": False}
    if output is not None:
        values = {f"{group}__{key}": array for group, scores in arrays.items() for key, array in scores.items()}
        values.update({f"{group}__active_plane": array for group, array in assignments.items()})
        np.savez_compressed(output / "per_example_metrics.npz", **values)
        write_json(output / "metrics.json", result)
    return result


def train(cfg):
    if min(cfg.width, cfg.steps, cfg.batch_size, cfg.val_per_group, cfg.eval_every, cfg.threads) < 1 or cfg.val_per_group % 4:
        raise ValueError("Training sizes/threads must be positive; validation group size must be divisible by four")
    if cfg.lr <= 0 or min(cfg.ssim_weight, cfg.pearson_weight, cfg.balanced_weight, cfg.concentration_weight) < 0:
        raise ValueError("Learning rate must be positive and loss weights nonnegative")
    if not cfg.diagnostic_only or not cfg.random_active_plane or cfg.active_plane is not None:
        raise ValueError("This script is a random-depth single-active-plane diagnostic only")
    if (cfg.model, cfg.phase_head, cfg.position, cfg.protocol) != ("transport_complex", "circular", "xy", PEAK_PROTOCOL):
        raise ValueError("The curriculum requires its declared transport_complex/v2 architecture")
    output = Path("outputs") / cfg.run
    if output.exists():
        raise FileExistsError(output)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(cfg.threads)
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
    config = {**asdict(cfg), "evaluation_planes": [0, 1, 2, 3], "joint_four_plane_evaluation": False,
              "selection_metric": "minimum balanced-depth single-plane validation objective",
              "scope": "Shared-weight pretraining only; direct export as a challenge model is forbidden.",
              "active_plane_seed": cfg.seed + 3000, "validation_active_plane_assignment": "row_index % 4 within each group",
              "parameters": sum(parameter.numel() for parameter in model.parameters()), "torch": torch.__version__,
              "amplitude_scales": [1.0] * 4, "test_used": False}
    write_json(output / "config.json", config)
    write_json(output / "validation_tuples.json", {key: records.tolist() for key, records in manifest.items()})
    write_json(output / "validation_active_planes.json", {key: (np.arange(len(records)) % 4).tolist() for key, records in manifest.items()})
    bank.save_manifest(output / "data_manifest.json", cfg.val_per_group)
    snapshot_source(output)
    write_json(output / "initial_metrics.json", evaluate_curriculum(model, bank, physics, manifest, cfg))
    data_rng = np.random.default_rng(cfg.seed + 1000)
    active_rng = np.random.default_rng(cfg.seed + 3000)
    best_objective = float("inf")
    training_seconds = 0.0
    for step in range(1, cfg.steps + 1):
        model.train()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        tick = time.perf_counter()
        targets = prepare_target(bank.fetch(bank.sample_train(cfg.batch_size, data_rng, cfg.domain), cfg.device), PEAK_PROTOCOL)
        planes = torch.as_tensor(active_rng.integers(0, 4, cfg.batch_size), device=cfg.device, dtype=torch.int64)
        optimizer.zero_grad(set_to_none=True)
        phase = model(masked_plane_inputs(targets, planes))
        terms = active_loss_terms(physics.amplitude(phase), targets, planes, cfg)
        loss = terms["objective"].mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite curriculum loss at step {step}")
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if cfg.device == "cuda":
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - tick
        if step == 1 or step % 50 == 0 or step == cfg.steps:
            entry = {"step": step, "diagnostic_only": True, "random_active_plane": True,
                     **{key: float(value.detach().mean()) for key, value in terms.items()},
                     "active_plane_counts": torch.bincount(planes, minlength=4).cpu().tolist(),
                     "grad_norm": float(gradient), "training_seconds": training_seconds}
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            print(json.dumps({"run": cfg.run, **entry}), flush=True)
        if step % cfg.eval_every == 0 or step == cfg.steps:
            validation = evaluate_curriculum(model, bank, physics, manifest, cfg)
            scores = validation["active_plane_aggregate"]
            with (output / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"step": step, "validation": validation}) + "\n")
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                          "step": step, "training_seconds": training_seconds, "diagnostic_only": True,
                          "random_active_plane": True, "evaluation_planes": [0, 1, 2, 3], "joint_four_plane_evaluation": False}
            torch.save(checkpoint, output / "last.pt")
            if scores["objective"] < best_objective:
                best_objective = scores["objective"]
                torch.save(checkpoint, output / "best.pt")
            print(json.dumps({"run": cfg.run, "step": step, "diagnostic_only": True,
                              "active_ssim": scores["ssim"], "active_pearson": scores["pearson"],
                              "active_objective": scores["objective"]}), flush=True)
    selected = torch.load(output / "best.pt", map_location=cfg.device, weights_only=False)
    model.load_state_dict(selected["model"])
    result = evaluate_curriculum(model, bank, physics, manifest, cfg, output=output)
    result.update({"selected_step": selected["step"], "selected_checkpoint": "best.pt", "training_seconds": training_seconds,
                   "total_seconds": time.perf_counter() - started, "selection_metric": config["selection_metric"]})
    write_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = Config()
    for key in ("run", "device", "domain"):
        parser.add_argument("--" + key.replace("_", "-"), default=getattr(defaults, key))
    for key in ("width", "steps", "batch_size", "seed", "data_seed", "val_per_group", "eval_every", "threads"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=getattr(defaults, key))
    parser.add_argument("--lr", type=float, default=defaults.lr)
    train(Config(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
