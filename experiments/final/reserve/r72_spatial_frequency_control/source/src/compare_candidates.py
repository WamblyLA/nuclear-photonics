"""Paired, stratified quartet bootstrap of saved evaluations; opens no images.

NPZ rows follow their adjacent tuple manifest, as written by src.train.evaluate.
The saved format has no row identifiers inside NPZ: this utility verifies the
manifest order, row counts, summaries and evaluator snapshot, not image content.
Intervals describe resampling of the observed quartets. Reused source images
mean they are not image-independent confidence intervals for a population.
"""

import argparse
import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METRICS = ("ssim", "pearson")
LEGACY_PROTOCOL = "fresnel_png_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


@dataclass
class Evaluation:
    directory: Path
    manifest: dict
    arrays: dict
    identity: dict
    provenance: dict


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Expected a nonempty group-to-quartet manifest")
    clean = {}
    for group, values in sorted(manifest.items()):
        records = np.asarray(values)
        if records.ndim != 3 or records.shape[1:] != (4, 2) or not len(records):
            raise ValueError(f"Invalid quartet shape for {group}")
        if not np.issubdtype(records.dtype, np.integer) or np.any(records < 0) or np.any(records[..., 0] > 2):
            raise ValueError(f"Invalid dataset/source indices for {group}")
        if any(len(set(map(tuple, row))) != 4 for row in records):
            raise ValueError(f"Sources must be distinct within each quartet in {group}")
        clean[group] = records.tolist()
    return clean


def evaluator_fingerprint(directory, hashes):
    """Verify saved code, ignoring training-only changes in train.py."""
    hashes = {name.replace("\\", "/"): value for name, value in hashes.items()}
    required = ("src/data.py", "src/metrics.py", "src/physics/fresnel.py", "src/train.py")
    keys = required + (("src/protocol.py",) if "src/protocol.py" in hashes else ())
    selected = {}
    for key in keys:
        path = directory / "source" / key
        if key not in hashes or not path.is_file() or file_sha256(path) != hashes[key]:
            raise ValueError(f"Missing or mismatched evaluator source snapshot: {key}")
        selected[key] = hashes[key]
    tree = ast.parse((directory / "source/src/train.py").read_text(encoding="utf-8-sig"))
    names = {"evaluate", "png_roundtrip", "image_metrics", "summarize"}
    functions = {node.name: ast.dump(node, include_attributes=False) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names}
    if functions.keys() != names:
        raise ValueError("Incomplete saved evaluation functions")
    selected["src/train.py"] = canonical_sha256(functions)
    return selected


def load_evaluation(directory, repo_root=REPO_ROOT):
    directory = Path(directory).resolve()
    repo_root = Path(repo_root).resolve()
    relative = directory.relative_to(repo_root).as_posix()
    files = [directory / "metrics.json", directory / "per_example_metrics.npz"]
    metrics = read_json(files[0])
    manifest_paths = [directory / name for name in ("validation_tuples.json", "test_tuples.json") if (directory / name).is_file()]
    if len(manifest_paths) != 1:
        raise ValueError("Exactly one validation/test tuple manifest is required")
    manifest_path = manifest_paths[0]
    files.append(manifest_path)
    manifest = validate_manifest(read_json(manifest_path))
    evaluation_path = directory / "evaluation_config.json"
    if evaluation_path.is_file():
        files.append(evaluation_path)
        evaluation = read_json(evaluation_path)
        config = evaluation["training_config"]
    else:
        config_path = directory / "config.json"
        files.append(config_path)
        config = read_json(config_path)
        evaluation = {}
    hashes_path = directory / "source_hashes.json"
    files.append(hashes_path)
    hashes = read_json(hashes_path)
    if "source_hashes" in evaluation and evaluation["source_hashes"] != hashes:
        raise ValueError("Conflicting source hashes in evaluation metadata")
    fingerprint = evaluator_fingerprint(directory, hashes)
    input_protocol = config.get("protocol", LEGACY_PROTOCOL)
    protocol = metrics["protocol"]
    for payload in (metrics, evaluation):
        if payload.get("model_input_protocol", input_protocol) != input_protocol:
            raise ValueError("Conflicting saved model-input protocols")
    if evaluation.get("evaluation_protocol", protocol) != protocol:
        raise ValueError("Conflicting metric protocols")
    test_used = manifest_path.name == "test_tuples.json"
    if metrics.get("test_used", test_used) != test_used:
        raise ValueError("Manifest split conflicts with test_used flag")
    source_identity = {"namespace": "official_training_files_validation_rows", "split_seed": config["data_seed"]}
    if test_used:
        path = directory / "test_provenance.json"
        files.append(path)
        source = read_json(path)
        if source["test_tuples_sha256"] != file_sha256(manifest_path):
            raise ValueError("Test manifest hash mismatch")
        source_identity = {"namespace": source["namespace"], "sources": source["sources"], "preprocessing": source["preprocessing"]}
    if metrics.get("source_namespace", source_identity["namespace"]) != source_identity["namespace"]:
        raise ValueError("Conflicting source namespaces")
    identity = {"protocol": protocol, "model_input_protocol": input_protocol, "source_identity": source_identity, "evaluator_fingerprint": fingerprint}
    if protocol == LEGACY_PROTOCOL:
        identity["amplitude_scales"] = config["amplitude_scales"]
    arrays = {}
    with np.load(directory / "per_example_metrics.npz", allow_pickle=False) as archive:
        for key in METRICS + ("pearson_valid",):
            found = {name.rsplit("__", 1)[0] for name in archive.files if name.endswith("__" + key)}
            if found != manifest.keys():
                raise ValueError(f"NPZ groups differ from manifest for {key}")
        for group, rows in manifest.items():
            arrays[group] = {}
            for key in METRICS + ("pearson_valid",):
                values = archive[f"{group}__{key}"]
                if values.shape != (len(rows), 4) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid metric rows: {group}/{key}")
                if key == "pearson_valid":
                    if values.dtype != np.bool_ or not values.all():
                        raise ValueError("Undefined Pearson cannot be silently dropped in a paired comparison")
                else:
                    arrays[group][key] = values.astype(np.float64)
                    recorded = metrics["groups"][group][key]
                    if recorded is None or not np.isclose(values.mean(dtype=np.float64), recorded, rtol=0, atol=1e-6):
                        raise ValueError(f"NPZ/group summary mismatch: {group}/{key}")
    total = sum(len(rows) for rows in manifest.values())
    if metrics["tuples"] != total or metrics["groups"].keys() != manifest.keys():
        raise ValueError("Metric summary and manifest group/count mismatch")
    for key in METRICS:
        average = np.concatenate([values[key] for values in arrays.values()]).mean()
        recorded = metrics["aggregate"][key]
        if recorded is None or not np.isclose(average, recorded, rtol=0, atol=1e-6):
            raise ValueError(f"NPZ/aggregate summary mismatch: {key}")
    provenance = {
        "directory": relative, "manifest_sha256": canonical_sha256(manifest),
        "manifest_hash_definition": "SHA256 of sorted-key compact UTF-8 JSON; row and plane order preserved",
        "files": [{"path": path.relative_to(repo_root).as_posix(), "sha256": file_sha256(path)} for path in files],
        "test_used": test_used, "checkpoint_sha256": evaluation.get("checkpoint_sha256", metrics.get("checkpoint_sha256")),
    }
    return Evaluation(directory, manifest, arrays, identity, provenance)


def validate_alignment(reference, candidate):
    if reference.identity != candidate.identity:
        differing = sorted(key for key in reference.identity.keys() | candidate.identity.keys() if reference.identity.get(key) != candidate.identity.get(key))
        raise ValueError(f"Incompatible evaluation identity: {', '.join(differing)}")
    if reference.manifest != candidate.manifest:
        raise ValueError("Quartet manifest mismatch: groups, source rows and plane order must match exactly")
    if reference.provenance["manifest_sha256"] != candidate.provenance["manifest_sha256"]:
        raise ValueError("Canonical manifest hash mismatch")


def paired_bootstrap(reference, candidate, repetitions=2000, seed=20261108, confidence=0.95):
    """Resample paired whole quartets within groups, jointly for both metrics."""
    if not isinstance(repetitions, int) or repetitions < 2:
        raise ValueError("repetitions must be an integer >= 2")
    if not isinstance(seed, int) or seed < 0 or not 0 < confidence < 1:
        raise ValueError("seed must be nonnegative and confidence in (0, 1)")
    if not reference or reference.keys() != candidate.keys():
        raise ValueError("Paired groups must match and be nonempty")
    rng = np.random.default_rng(seed)
    total = sum(len(values["ssim"]) for values in reference.values())
    pooled_draws = np.zeros((repetitions, len(METRICS)), dtype=np.float64)
    pooled_reference = np.zeros(len(METRICS))
    pooled_candidate = np.zeros(len(METRICS))
    alpha = (1 - confidence) / 2

    def describe(left, right, draws, count):
        bounds = np.quantile(draws, [alpha, 1 - alpha], axis=0, method="linear")
        return {"quartets": count, "planes": 4 * count, "metrics": {
            metric: {"reference_mean": float(left[j]), "candidate_mean": float(right[j]),
                     "difference": float(right[j] - left[j]), "ci_low": float(bounds[0, j]),
                     "ci_high": float(bounds[1, j]), "bootstrap_std": float(draws[:, j].std(ddof=1))}
            for j, metric in enumerate(METRICS)
        }}

    groups = {}
    for group in sorted(reference):
        values = []
        for source in (reference[group], candidate[group]):
            selected = [np.asarray(source[metric], dtype=np.float64) for metric in METRICS]
            if any(value.ndim != 2 or value.shape[1] != 4 or not len(value) or not np.isfinite(value).all() for value in selected) or selected[0].shape != selected[1].shape:
                raise ValueError(f"Expected finite [quartets, 4] metrics in {group}")
            values.append(np.stack(selected, axis=-1).mean(axis=1))
        left, right = values
        if left.shape != right.shape:
            raise ValueError(f"Paired row counts differ in {group}")
        count = len(left)
        differences = right - left
        draws = np.empty((repetitions, len(METRICS)), dtype=np.float64)
        # Bounded allocation also supports much larger manifests than tonight's.
        for start in range(0, repetitions, 128):
            stop = min(start + 128, repetitions)
            indices = rng.integers(count, size=(stop - start, count))
            draws[start:stop] = differences[indices].mean(axis=1)
        groups[group] = describe(left.mean(axis=0), right.mean(axis=0), draws, count)
        weight = count / total
        pooled_draws += weight * draws
        pooled_reference += weight * left.mean(axis=0)
        pooled_candidate += weight * right.mean(axis=0)
    return {"aggregate": describe(pooled_reference, pooled_candidate, pooled_draws, total), "groups": groups}


def compare_evaluations(reference, candidates, repetitions=2000, seed=20261108, confidence=0.95):
    if not candidates:
        raise ValueError("At least one candidate is required")
    comparisons = []
    for candidate in candidates:
        validate_alignment(reference, candidate)
        result = paired_bootstrap(reference.arrays, candidate.arrays, repetitions, seed, confidence)
        comparisons.append({"candidate": candidate.provenance, **result})
    return {
        "reference": reference.provenance, "shared_evaluation_identity": reference.identity,
        "method": {"name": "paired_stratified_quartet_percentile_bootstrap", "repetitions": repetitions,
                   "seed": seed, "confidence": confidence, "numpy": np.__version__,
                   "difference": "candidate minus reference; positive is better",
                   "resampling": "whole quartets within each group; fixed original group sizes; identical draws for both metrics and every candidate",
                   "aggregate_weighting": "each quartet has equal weight; four planes remain together",
                   "interpretation": "Intervals are conditional on the empirical distribution of sampled quartets. Sources can repeat across quartets/groups; these are not independent-source or population confidence intervals.",
                   "multiple_comparisons": "No multiplicity correction; descriptive exploratory intervals, not adjusted hypothesis tests",
                   "alignment_limit": "NPZ has no embedded source IDs; rows are assumed to follow the verified adjacent manifest and saved evaluator."},
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()
    reference = load_evaluation(args.reference)
    candidates = [load_evaluation(path) for path in args.candidates]
    result = compare_evaluations(reference, candidates, args.repetitions, args.seed, args.confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparisons": len(candidates), "quartets": sum(len(rows) for rows in reference.manifest.values())}))


if __name__ == "__main__":
    main()
