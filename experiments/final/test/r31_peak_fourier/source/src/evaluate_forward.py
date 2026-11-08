import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import PIL
import torch

from src.curriculum import sample_forward_targets
from src.models.networks import build_model
from src.physics.fresnel import FresnelPropagator, phase_to_uint8, uint8_to_phase
from src.protocol import LEGACY_PROTOCOL
from src.train import image_metrics, png_roundtrip, summarize, write_json


@torch.no_grad()
def make_targets(samples, batch_size, propagator, scales, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    targets, statistics = [], []
    for start in range(0, samples, batch_size):
        target, stats = sample_forward_targets(min(batch_size, samples - start), propagator, scales, generator)
        codes = (target * 255).round().to(torch.uint8)
        decoded = png_roundtrip(codes).float() / 255
        assert torch.equal(target, decoded)
        targets.append(decoded)
        statistics.append((len(target), stats))
    pooled = {key: sum(size * values[key] for size, values in statistics) / samples for key in statistics[0][1]}
    return torch.cat(targets), pooled


def initialize_model(config):
    torch.manual_seed(config["seed"])
    model = build_model(config["model"], config["width"], config["phase_head"], config["position"])
    if config["model"] != "interleave":
        torch.manual_seed(config["seed"])
        reference = build_model("cnn", config["width"], config["phase_head"])
        model.load_state_dict(reference.state_dict(), strict=False)
    return model.eval()


@torch.no_grad()
def evaluate(model, targets, propagator, scales, batch_size):
    batches = {}
    for start in range(0, len(targets), batch_size):
        target = targets[start:start + batch_size]
        phase = model(target)
        decoded_phase = uint8_to_phase(png_roundtrip(phase_to_uint8(phase)))
        raw = propagator.amplitude(decoded_phase) * scales
        reconstruction = png_roundtrip((raw.clamp(0, 1) * 255).round().to(torch.uint8)).float() / 255
        values = image_metrics(reconstruction, target)
        values["raw_mse"] = (raw - target).square().mean((-2, -1))
        values["clipped_fraction"] = (raw > 1).float().mean((-2, -1))
        for key, value in values.items():
            batches.setdefault(key, []).append(value.numpy())
    return summarize({key: np.concatenate(value) for key, value in batches.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=99042)
    parser.add_argument("--output", type=Path, default=Path("experiments/cycle2_forward_generalization.json"))
    parser.add_argument("--include-untrained", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.samples, args.batch_size) < 1:
        raise ValueError("samples and batch_size must be positive")
    torch.set_num_threads(4)
    configs = [json.loads((run / "config.json").read_text(encoding="utf-8")) for run in args.runs]
    if any(config.get("protocol", LEGACY_PROTOCOL) != LEGACY_PROTOCOL for config in configs):
        raise ValueError("This historical synthetic diagnostic implements only fresnel_png_v1; do not use it to score v2 checkpoints")
    if any(config["amplitude_scales"] != configs[0]["amplitude_scales"] for config in configs[1:]):
        raise ValueError("All models must use the same amplitude scales")
    if any(config["seed"] + 2000 == args.seed for config in configs):
        raise ValueError("Synthetic evaluation seed must differ from training seed")
    propagator = FresnelPropagator()
    scales = torch.tensor(configs[0]["amplitude_scales"]).reshape(1, 4, 1, 1)
    targets, stats = make_targets(args.samples, args.batch_size, propagator, scales, args.seed)
    repeated, repeated_stats = make_targets(args.samples, args.batch_size, propagator, scales, args.seed)
    assert torch.equal(targets, repeated) and stats == repeated_stats
    target_codes = (targets * 255).round().to(torch.uint8).numpy()
    result = {
        "device": "cpu", "torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__,
        "seed": args.seed, "samples": args.samples, "batch_size": args.batch_size,
        "amplitude_scales": configs[0]["amplitude_scales"],
        "target_uint8_sha256": hashlib.sha256(target_codes.tobytes()).hexdigest(),
        "generation_repeat_identical": True, "generation_statistics": stats,
        "actual_target_rms": float(targets.square().mean().sqrt()),
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), Path("src/curriculum.py"), Path("src/models/networks.py"), Path("src/metrics.py"), Path("src/physics/fresnel.py"), Path("src/train.py"))},
        "scope": "New synthetic forward-only targets, generated on CPU with a distinct seed. No real validation/test data or phase labels. Same saved-PNG evaluator. Synthetic raw fields are physically realizable before clipping/rounding; clipped PNG targets need not be exactly realizable. CPU and CUDA random streams are not identical.",
        "runs": [],
    }
    if args.include_untrained:
        metrics = evaluate(initialize_model(configs[0]), targets, propagator, scales, args.batch_size)
        result["runs"].append({"run": "untrained_" + args.runs[0].name, "training_steps": 0, "initialization_config": configs[0], "metrics": metrics})
    for run, config in zip(args.runs, configs):
        checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model = initialize_model(config)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate(model, targets, propagator, scales, args.batch_size)
        result["runs"].append({"run": run.name, "training_steps": config["steps"], "selected_step": checkpoint["step"], "synthetic_steps": config.get("synthetic_steps", 0), "checkpoint_sha256": hashlib.sha256((run / "best.pt").read_bytes()).hexdigest(), "metrics": metrics})
    write_json(args.output, result)
    for run in result["runs"]:
        print(json.dumps({"run": run["run"], **{key: run["metrics"][key] for key in ("ssim", "pearson", "mse", "raw_mse")}}), flush=True)
    print(json.dumps({"target_statistics": stats, "target_sha256": result["target_uint8_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
