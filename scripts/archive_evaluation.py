"""Package a saved evaluation without images, datasets or model weights."""

import argparse
import hashlib
import json
import math
import stat
from pathlib import Path, PurePosixPath

import numpy as np

from scripts.archive_research import PROJECT_ROOT, public_values
from src.compare_candidates import load_evaluation


METADATA_FILES = ("evaluation_config.json", "manifest_provenance.json", "test_provenance.json", "frozen_model.json")
EXACT_FILES = ("metrics.json", "per_example_metrics.npz", "source_hashes.json")


def reject_links(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Parent traversal is not allowed")
    for item in (path, *path.parents):
        try:
            attributes = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError("Symlinks and filesystem reparse points are not allowed")
    return path.resolve()


def snapshot_name(value):
    if not isinstance(value, str):
        raise ValueError("Source snapshot paths must be strings")
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value or len(path.parts) < 2 or path.parts[0] != "src" or path.suffix != ".py":
        raise ValueError("Expected a relative src/*.py snapshot path")
    return "source/" + path.as_posix()


def finite_json(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite JSON value cannot be archived")


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    finite_json(value)
    return value


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def archive_evaluation(input_dir, output_dir, project_root=PROJECT_ROOT):
    source, output = reject_links(input_dir), reject_links(output_dir)
    if output.exists():
        raise FileExistsError("Archive output already exists")
    if not source.is_dir() or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Input must be an existing, separate evaluation directory")
    manifests = [name for name in ("validation_tuples.json", "test_tuples.json") if (source / name).exists()]
    if len(manifests) != 1:
        raise ValueError("Exactly one validation/test tuple manifest is required")
    required = [*EXACT_FILES, manifests[0], "evaluation_config.json"]
    if manifests[0] == "test_tuples.json":
        required.append("test_provenance.json")
    for name in required:
        path = reject_links(source / name)
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluation file is missing: {name}")
    hashes = read_json(source / "source_hashes.json")
    snapshot_paths = {snapshot_name(name): digest for name, digest in hashes.items()}
    if not hashes or len(snapshot_paths) != len(hashes):
        raise ValueError("Missing or duplicate normalized source snapshot paths")
    snapshot_root = reject_links(source / "source")
    actual_sources = set()
    for path in snapshot_root.rglob("*"):
        reject_links(path)
        if path.is_file() and path.suffix == ".py":
            actual_sources.add(path.relative_to(source).as_posix())
    if actual_sources != snapshot_paths.keys():
        raise ValueError("Source snapshots and recorded hashes differ")
    for name, digest in snapshot_paths.items():
        if sha256((source / name).read_bytes()) != digest:
            raise ValueError(f"Source snapshot hash mismatch: {name}")
    names = sorted({*EXACT_FILES, manifests[0], *snapshot_paths, *(name for name in METADATA_FILES if (source / name).exists())})
    original = {}
    for name in names:
        path = reject_links(source / name)
        original[name] = path.read_bytes()
        if path.suffix == ".json":
            read_json(path)
    with np.load(source / "per_example_metrics.npz", allow_pickle=False) as arrays:
        if not arrays.files:
            raise ValueError("Per-example metrics are empty")
        for name in arrays.files:
            values = arrays[name]
            if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                raise ValueError(f"Nonfinite or nonnumeric metric array: {name}")
    # Verifies row counts, summaries, source fingerprint, split and protocol without loading images.
    evaluation = load_evaluation(source, repo_root=source.parent)
    redactions, payload, records = [], {}, []
    for name, data in original.items():
        transformation = "exact_bytes"
        if name in METADATA_FILES:
            value = json.loads(data.decode("utf-8-sig"))
            clean = public_values(value, Path(project_root), redactions)
            if "source_hashes" in value:
                if value["source_hashes"] != hashes:
                    raise ValueError("Conflicting source hashes in metadata")
                # Preserve safe relative Windows keys: comparison expects this exact mapping.
                clean["source_hashes"] = hashes
            data = (json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
            transformation = "public_values_paths_and_json_formatting; source_hashes_mapping_preserved"
        payload[name] = data
        records.append({"path": name, "original_sha256": sha256(original[name]), "archived_sha256": sha256(data), "original_bytes": len(original[name]), "archived_bytes": len(data), "transformation": transformation})
    for name, data in original.items():
        if reject_links(source / name).read_bytes() != data:
            raise RuntimeError("Evaluation changed during archive preparation")
    index = {
        "format_version": 1,
        "source_evaluation": public_values(str(source), Path(project_root), redactions),
        "protocol": evaluation.identity["protocol"],
        "test_used": evaluation.provenance["test_used"],
        "files": records,
        "external_path_redactions": redactions,
        "transformation": "Numerical files, tuple manifest, source_hashes.json and Python snapshots retain exact bytes. Only listed metadata JSON receives public_values path sanitization and UTF-8 JSON formatting. Safe relative source_hashes keys retain their original spelling for comparison compatibility.",
        "excluded": ["checkpoints", "images", "examples", "datasets", "unlisted files"],
    }
    reject_links(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payload.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    (output / "archive_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = archive_evaluation(args.input, args.output)
    print(json.dumps({"files": len(result["files"]), "protocol": result["protocol"], "test_used": result["test_used"]}))


if __name__ == "__main__":
    main()
