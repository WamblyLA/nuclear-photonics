"""Select five reproducible demonstration sets by labels, never by model scores."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import ImageBank


def prepare_examples(manifest_path, output, count=5):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if count < 5:
        raise ValueError("The submission needs at least five demonstration sets")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bank = ImageBank(seed=42)
    selected = []
    for row in manifest["mixed"]:
        objects = [(int(dataset), int(bank.labels[dataset][index])) for dataset, index in row]
        if len(set(objects)) != 4:
            continue
        if any(index not in bank.val[dataset] for dataset, index in row):
            raise ValueError("Demonstrations must come from validation sources")
        selected.append((row, objects))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Not enough distinct-class quartets in this manifest")
    output.mkdir(parents=True)
    records = []
    for number, (row, objects) in enumerate(selected, start=1):
        folder = output / f"input_set{number}"
        folder.mkdir()
        images = bank.fetch(np.asarray([row]))[0]
        files = []
        for plane, (array, source, object_key) in enumerate(zip(images, row, objects), start=1):
            path = folder / f"image{plane}.png"
            Image.fromarray((255 * array).round().to(dtype=torch.uint8).numpy()).save(path)
            files.append({"path": path.relative_to(output).as_posix(), "dataset": bank.names[source[0]], "train_source_index": source[1], "class_label": object_key[1], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        records.append({"set": number, "inputs": files})
    provenance = {"selection_rule": "First five existing mixed selection-validation quartets with four distinct (dataset, class) pairs; no model scores used", "split_seed": 42, "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "split": "selection_validation", "test_used": False, "sets": records}
    (output / "manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sets": len(records), "directory": str(output), "test_used": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("examples"))
    args = parser.parse_args()
    prepare_examples(args.manifest, args.output)
