"""List saved checkpoints passing the recorded selection Pearson guard."""

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PEAK_PROTOCOL = "fresnel_png_v2_peak"
CHECKPOINT_ALIASES = ("best.pt", "best_pearson.pt", "last.pt")
INFERENCE_FIELDS = ("run", "model", "width", "phase_head", "position", "protocol", "amplitude_scales")


@dataclass
class Config:
    runs: list
    output: Path
    policy: Path = Path("experiments/night_selection_policy.json")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_path(path, root):
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("All research inputs and reports must be inside the repository") from error


def checkpoint_metadata(path):
    import torch
    # Trusted local research artifacts only. mmap avoids materializing model/optimizer tensors.
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return {"step": saved["step"], "config": saved["config"]}


def diagnostic(config, metrics):
    return (
        config.get("fixed_train_quartets", 0) > 0
        or config.get("diagnostic_only", False)
        or config.get("training_only_diagnostic", False)
        or metrics.get("diagnostic_only", False)
        or metrics.get("training_only_diagnostic", False)
        or str(config.get("run", "")).startswith("smoke_")
    )


def available_checkpoints(folder, eligible_steps):
    paths = []
    for path in folder.glob("*.pt"):
        match = re.fullmatch(r"step_(\d+)\.pt", path.name)
        if path.name in CHECKPOINT_ALIASES or (match and int(match.group(1)) in eligible_steps):
            paths.append(path)
    # Prefer the explicit numbered snapshot, then a stable named checkpoint.
    return sorted(paths, key=lambda path: (0 if path.name.startswith("step_") else 1, path.name))


def collect_shortlist(config, project_root=PROJECT_ROOT, metadata_reader=checkpoint_metadata):
    output, policy_path = Path(config.output), Path(config.policy)
    if output.exists():
        raise FileExistsError("Shortlist output already exists; use a new report filename")
    relative_path(output, project_root)
    folders = sorted({Path(path).resolve() for path in config.runs}, key=lambda path: path.name)
    for folder in folders:
        relative_path(folder, project_root)
        if output.resolve().is_relative_to(folder):
            raise ValueError("The report must not be written inside an original research run")
    policy = load_json(policy_path)
    threshold = policy["minimum_selection_mean_pearson"]
    if policy["protocol"] != PEAK_PROTOCOL or type(threshold) not in (int, float) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Expected a finite Pearson guard for fresnel_png_v2_peak")
    candidates, excluded_runs, rejected_checkpoints, missing_steps = [], [], [], []
    common_manifest_sha = None
    accepted_runs = []
    for folder in folders:
        run_name = folder.name
        required = ("config.json", "metrics.json", "history.jsonl", "validation_tuples.json")
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            excluded_runs.append({"run": run_name, "reason": "run is incomplete", "missing_files": missing})
            continue
        run_config, final_metrics = load_json(folder / "config.json"), load_json(folder / "metrics.json")
        if diagnostic(run_config, final_metrics):
            excluded_runs.append({"run": run_name, "reason": "training-only diagnostic; excluded from release"})
            continue
        if run_config.get("protocol") != PEAK_PROTOCOL or final_metrics.get("protocol", run_config["protocol"]) != PEAK_PROTOCOL:
            excluded_runs.append({"run": run_name, "reason": "not the peak-normalized v2 protocol"})
            continue
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if max((row.get("step", 0) for row in history), default=0) < run_config["steps"]:
            excluded_runs.append({"run": run_name, "reason": "history does not reach the configured training budget"})
            continue
        manifest_sha = file_hash(folder / "validation_tuples.json")
        if common_manifest_sha is None:
            common_manifest_sha = manifest_sha
        elif manifest_sha != common_manifest_sha:
            raise ValueError("Cannot rank checkpoints evaluated on different validation manifests")
        accepted_runs.append(run_name)
        validation = {}
        for row in history:
            if "validation" not in row:
                continue
            scores = row["validation"]
            if scores.get("protocol", run_config["protocol"]) != PEAK_PROTOCOL:
                raise ValueError("A run contains mixed-protocol validation history")
            mean = scores["aggregate"]
            pearson, ssim = mean.get("pearson"), mean.get("ssim")
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in (pearson, ssim)):
                continue
            if pearson < threshold or mean.get("invalid_pearson_count", 0) > 0:
                continue
            step = row["step"]
            if type(step) is not int or step < 0:
                raise ValueError("Validation step must be a nonnegative integer")
            if step in validation and validation[step]["validation"] != scores:
                raise ValueError("Conflicting validation results are recorded for the same step")
            validation[step] = row
        if not validation:
            continue
        by_step = {}
        for path in available_checkpoints(folder, validation):
            public_path = relative_path(path, project_root)
            metadata = metadata_reader(path)
            step = metadata["step"]
            embedded = metadata["config"]
            match = re.fullmatch(r"step_(\d+)\.pt", path.name)
            reason = None
            if type(step) is not int or step < 0:
                reason = "invalid embedded checkpoint step"
            elif match and step != int(match.group(1)):
                reason = "filename step differs from embedded checkpoint step"
            elif diagnostic(embedded, {}) or embedded.get("protocol") != PEAK_PROTOCOL:
                reason = "embedded checkpoint is diagnostic or uses a different protocol"
            elif any(embedded.get(name) != run_config.get(name) for name in INFERENCE_FIELDS):
                reason = "embedded checkpoint configuration differs from its research run"
            if reason:
                rejected_checkpoints.append({"checkpoint": public_path, "reason": reason})
                continue
            if step not in validation:
                continue
            by_step.setdefault(step, []).append({"path": public_path, "sha256": file_hash(path), "bytes": path.stat().st_size})
        for step, row in validation.items():
            if step not in by_step:
                missing_steps.append({"run": run_name, "step": step, "ssim": row["validation"]["aggregate"]["ssim"], "pearson": row["validation"]["aggregate"]["pearson"], "reason": "no verified saved checkpoint remains for this eligible validation step"})
                continue
            artifacts = by_step[step]
            candidates.append({
                "run": run_name, "model": run_config["model"], "width": run_config["width"],
                "phase_head": run_config["phase_head"], "position": run_config.get("position", "none"),
                "protocol": PEAK_PROTOCOL, "step": step,
                "checkpoint": artifacts[0]["path"], "checkpoint_sha256": artifacts[0]["sha256"],
                "verified_same_step_aliases": artifacts,
                "selection_validation": row["validation"],
                "training_seconds_at_step": row.get("training_seconds"),
                "requested_run_steps": run_config["steps"], "validation_manifest_sha256": manifest_sha,
                "config_sha256": file_hash(folder / "config.json"), "history_sha256": file_hash(folder / "history.jsonl"),
                "pearson_guard_passed": True, "foreground_guard": "not evaluated", "visual_review": "not performed",
            })
    candidates.sort(key=lambda item: (-item["selection_validation"]["aggregate"]["ssim"], -item["selection_validation"]["aggregate"]["pearson"], item["run"], item["step"]))
    result = {
        "schema_version": 1, "winner_selected": False,
        "scope": "Recorded selection-validation history of completed v2 runs; no model inference, reserve evaluation or official test data are opened",
        "selection_policy": {"path": relative_path(policy_path, project_root), "sha256": file_hash(policy_path), "minimum_selection_mean_pearson": threshold, "minimum_mnist_foreground_brightness_ratio": policy.get("minimum_mnist_foreground_brightness_ratio")},
        "ordering": "Local SSIM descending, Pearson descending as a tie-breaker, then run and step; one entry per verified run/step",
        "remaining_checks": ["foreground brightness guard on the specified diagnostic examples", "visual quality and finite outputs", "source-disjoint validation before freezing one model", "official test only after freezing; never for checkpoint selection"],
        "checkpoint_alias_policy": "Aliases are verified by embedded run/config/step; they need not have identical serialized bytes. The first listed artifact is preferred, with numbered snapshots before named aliases.",
        "common_validation_manifest_sha256": common_manifest_sha,
        "accepted_runs": accepted_runs, "candidates": candidates,
        "excluded_runs": excluded_runs, "rejected_checkpoints": rejected_checkpoints,
        "eligible_steps_without_saved_checkpoint": missing_steps,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("experiments/night_selection_policy.json"))
    args = parser.parse_args()
    result = collect_shortlist(Config(**vars(args)))
    print(json.dumps({"candidate_checkpoints": len(result["candidates"]), "completed_runs": len(result["accepted_runs"]), "excluded_runs": len(result["excluded_runs"]), "rejected_checkpoints": len(result["rejected_checkpoints"]), "winner_selected": False}))


if __name__ == "__main__":
    main()
