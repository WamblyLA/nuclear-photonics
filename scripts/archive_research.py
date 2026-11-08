"""Archive completed research runs as public, checkpoint-free provenance."""

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ("fresnel_png_v1", "fresnel_png_v2_peak")
REQUIRED_FILES = ("config.json", "metrics.json", "source_hashes.json", "validation_tuples.json", "history.jsonl", "best.pt")


@dataclass
class Config:
    runs: Path = Path("outputs")
    output: Path = Path("experiments/run_registry.json")
    include_training_samples: bool = False


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def public_values(value, root, redactions, field=""):
    """Keep scientific values intact while making local filesystem paths portable."""
    if isinstance(value, dict):
        return {public_values(key, root, redactions, field): public_values(item, root, redactions, f"{field}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [public_values(item, root, redactions, f"{field}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, str):
        candidate = value.replace("\\", "/")
        root_name = str(root.resolve()).replace("\\", "/").rstrip("/")
        if candidate.lower().startswith(root_name.lower() + "/"):
            return candidate[len(root_name) + 1:]
        if re.match(r"^(?:[A-Za-z]:/|/)", candidate):
            label = f"external_path_redacted_{len(redactions) + 1}"
            redactions.append(label)
            return label
        return candidate if "\\" in value else value
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite scientific value cannot be silently archived")
    return value


def scalar_metrics(metrics):
    return {name: value for name, value in metrics.items() if value is None or isinstance(value, (int, float, bool))}


def validation_point(step, result, seconds, protocol):
    result_protocol = result.get("protocol", protocol)
    if result_protocol != protocol:
        raise ValueError("A run mixes evaluation protocols in its validation history")
    return {
        "step": step,
        "training_seconds": seconds,
        "protocol": protocol,
        "aggregate": scalar_metrics(result["aggregate"]),
        "groups": {name: scalar_metrics(scores) for name, scores in result["groups"].items()},
    }


def training_summary(rows):
    names = sorted({name for row in rows for name, value in row.items() if type(value) in (int, float) and name != "step"})
    values = {}
    for name in names:
        observed = [row[name] for row in rows if name in row and type(row[name]) in (int, float)]
        values[name] = {"first": observed[0], "last": observed[-1], "min": min(observed), "max": max(observed), "logged_samples": len(observed)}
    return {"logged_rows": len(rows), "scope": "Sparse training log samples, not an average over all optimizer steps", "scalars": values}


def selected_step(folder, metrics):
    if "selected_step" in metrics:
        return int(metrics["selected_step"]), "metrics.json"
    # Older local research checkpoints omitted this value from metrics.json.
    # No model is evaluated; tensors are mapped to CPU and discarded immediately.
    import torch
    saved = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    return int(saved["step"]), "best.pt metadata (trusted local research artifact, CPU load)"


def archive_run(folder, project_root=PROJECT_ROOT, include_training_samples=False):
    config = load_json(folder / "config.json")
    metrics = load_json(folder / "metrics.json")
    manifest = load_json(folder / "validation_tuples.json")
    sources = load_json(folder / "source_hashes.json")
    history = [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    protocol = metrics.get("protocol", config.get("protocol", PROTOCOLS[0]))
    protocol_source = "metrics.json" if "protocol" in metrics else "config.json" if "protocol" in config else "historical v1 default"
    if "protocol" in config and config["protocol"] != protocol:
        raise ValueError("Checkpoint configuration and final evaluation use different protocols")
    config["protocol"] = protocol
    completed_step = max((row.get("step", 0) for row in history), default=0)
    if completed_step < config["steps"]:
        raise ValueError("History does not reach the requested training budget")
    chosen_step, step_source = selected_step(folder, metrics)
    if not 0 <= chosen_step <= completed_step:
        raise ValueError("Selected checkpoint step lies outside the completed run")
    validation = []
    initial_path = folder / "initial_metrics.json"
    if initial_path.is_file():
        validation.append(validation_point(0, load_json(initial_path), 0.0, protocol))
    validation.extend(validation_point(row["step"], row["validation"], row.get("training_seconds"), protocol) for row in history if "validation" in row)
    training = [row for row in history if "loss" in row and "validation" not in row]
    diagnostic = bool(config.get("diagnostic_only", False)) or config.get("fixed_train_quartets", 0) > 0
    candidate = not diagnostic and protocol in PROTOCOLS
    manifest_counts = {name: len(rows) for name, rows in manifest.items()}
    checkpoint_sha = file_hash(folder / "best.pt")
    result = {
        "run": folder.name,
        "status": "completed",
        "protocol": protocol,
        "protocol_source": protocol_source,
        "diagnostic_only": diagnostic,
        "diagnostic_reason": "Explicit diagnostic_only configuration or fixed TRAIN memorization check; not a challenge candidate" if diagnostic else None,
        "candidate_eligible": candidate,
        "candidate_scope": "Eligible for review within this protocol only; not a winner or quality endorsement",
        "config": config,
        "selected_step": chosen_step,
        "selected_step_source": step_source,
        "best_checkpoint": {"path": (folder / "best.pt").as_posix(), "sha256": checkpoint_sha, "bytes": (folder / "best.pt").stat().st_size, "included_in_public_archive": False},
        "budget": {
            "requested_steps": config["steps"],
            "completed_step": completed_step,
            "batch_size": config["batch_size"],
            "training_seconds": metrics.get("training_seconds"),
            "recorded_total_seconds": metrics.get("total_seconds"),
            "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
            "inference_ms_batch1": metrics.get("inference_ms_batch1"),
            "scope": "This run only; any warm-start source budget is recorded separately in its own run",
        },
        "metrics": metrics,
        "validation_manifest": {
            "path": (folder / "validation_tuples.json").as_posix(),
            "sha256": file_hash(folder / "validation_tuples.json"),
            "split": "selection_validation", "test_used": False,
            "quartets_per_group": manifest_counts, "total_quartets": sum(manifest_counts.values()),
            "dataset_order": ["CIFAR10", "MNIST", "FashionMNIST"],
            "tuple_schema": "group -> quartet -> four [dataset_id, original_train_index] pairs",
            "tuples_included": False,
        },
        "source_hashes": sources,
        "source_file_sha256": {name: file_hash(folder / name) for name in REQUIRED_FILES if name != "best.pt"},
        "validation_curve": validation,
        "training_log_summary": training_summary(training),
    }
    if include_training_samples:
        result["training_log_samples"] = [{name: value for name, value in row.items() if value is None or isinstance(value, (int, float, bool, str))} for row in training]
    probes = [row for row in history if "training_probe" in row]
    if probes:
        result["training_probe_curve"] = [{"step": row["step"], "training_only_diagnostic": True, "metrics": row["training_probe"]} for row in probes]
    redactions = []
    result = public_values(result, project_root, redactions)
    result["external_path_redactions"] = len(redactions)
    return result


def archive_research(config, project_root=PROJECT_ROOT):
    runs, output = Path(config.runs).resolve(), Path(config.output).resolve()
    if output.is_relative_to(runs):
        raise ValueError("Public registry must be outside original run outputs")
    records, excluded = [], []
    folders = sorted((path for path in runs.iterdir() if path.is_dir() and re.fullmatch(r"r\d+_.+", path.name)), key=lambda path: (int(re.match(r"r(\d+)", path.name).group(1)), path.name))
    for folder in folders:
        missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
        if missing:
            excluded.append({"run": folder.name, "status": "not_archived", "reason": "required completion artifacts are absent", "missing_files": missing})
            continue
        try:
            records.append(archive_run(folder, project_root, config.include_training_samples))
        except (ValueError, KeyError, OSError) as error:
            # Preserve a visible exclusion rather than turning malformed/in-progress data into results.
            message = error.strerror if isinstance(error, OSError) else str(error)
            excluded.append({"run": folder.name, "status": "not_archived", "reason": type(error).__name__ + ": " + message})
    protocol_groups = {}
    for protocol in sorted({record["protocol"] for record in records}):
        group = [record for record in records if record["protocol"] == protocol]
        protocol_groups[protocol] = {
            "runs": [record["run"] for record in group],
            "candidate_runs": [record["run"] for record in group if record["candidate_eligible"]],
            "diagnostic_runs": [record["run"] for record in group if record["diagnostic_only"]],
            "recorded_training_seconds": sum(record["budget"]["training_seconds"] or 0 for record in group),
        }
    result = {
        "schema_version": 1,
        "scope": "Completed local research runs; selection-validation results only. No official test data, raw images, checkpoints, or private chat are included.",
        "comparison_policy": "Compare within protocol and matched validation manifest/budget. fresnel_png_v1 and fresnel_png_v2_peak are not directly comparable. Candidate membership is not model selection; no winner is declared.",
        "learning_curve_policy": "Initial and periodic selection-validation scalar summaries are preserved. Final aggregate/group metrics also retain per-plane arrays. Training summaries describe logged batches only; training probes are separate diagnostics.",
        "archive_limits": "Hashes identify local original artifacts. Per-example NPZ arrays, validation tuples and source snapshots are not copied into this registry. The final selected evaluation will be archived separately.",
        "generator": {"path": "scripts/archive_research.py", "sha256": file_hash(Path(__file__))},
        "completed_runs": len(records),
        "diagnostic_runs": [record["run"] for record in records if record["diagnostic_only"]],
        "protocols": protocol_groups,
        "runs": records,
        "not_archived": excluded,
    }
    redactions = []
    result = public_values(result, project_root, redactions)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError("A partial registry already exists; inspect it before retrying")
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("experiments/run_registry.json"))
    parser.add_argument("--include-training-samples", action="store_true")
    args = parser.parse_args()
    result = archive_research(Config(**vars(args)))
    print(json.dumps({"output": str(args.output), "completed_runs": result["completed_runs"], "diagnostic_runs": result["diagnostic_runs"], "not_archived": len(result["not_archived"]), "protocol_counts": {name: len(group["runs"]) for name, group in result["protocols"].items()}}))


if __name__ == "__main__":
    main()
