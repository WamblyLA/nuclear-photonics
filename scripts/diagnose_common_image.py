"""Score one shared mean target image: no DOE, propagation, model or optimizer.

This abstract reference measures what the metrics reward without depth
separation. No physical realizability is established; it is not a candidate
or an upper bound. Only the previously used selection sources are inspected.
Run from the repository root: python -m scripts.diagnose_common_image
"""

import argparse
import hashlib
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import PIL
from PIL import Image
import torch

from src.data import ImageBank, NAMES, PREPROCESS_VERSION
from src.metrics import pearson, ssim
from src.protocol import PEAK_PROTOCOL, prepare_reconstruction, prepare_target


@dataclass
class Config:
    r49_run: Path = Path("outputs/r49_peak_complex_guard_continue")
    r56_run: Path = Path("outputs/r56_long_complex_w32")
    output: Path = Path("experiments/common_image_diagnostic.json")
    batch_size: int = 16
    threads: int = 2
    history_step: int | None = None


def read_source(path):
    payload = path.read_bytes()
    relative = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    return payload, {"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def latest_validation(path, step=None):
    payload, source = read_source(path)
    entries = []
    lines = payload.splitlines()
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                continue  # A running trainer may be appending the final line.
            raise
        if "validation" in row and (step is None or row["step"] == step):
            entries.append((row, number, line))
    if not entries:
        raise ValueError("No matching completed validation row")
    row, number, line = entries[-1]
    source.update({"line": number, "row_sha256": hashlib.sha256(line).hexdigest(), "step": row["step"]})
    return row["validation"], source


def common_image(target):
    mean = target.mean(dim=1, keepdim=True)
    codes = (prepare_reconstruction(mean, PEAK_PROTOCOL) * 255).round().to(torch.uint8)
    decoded = []
    for values in codes[:, 0].numpy():
        buffer = io.BytesIO()
        Image.fromarray(values).save(buffer, format="PNG")
        buffer.seek(0)
        with Image.open(buffer) as image:
            decoded.append(np.array(image, dtype=np.uint8))
    common = torch.from_numpy(np.stack(decoded)).float()[:, None] / 255
    return common.expand_as(target)


def summarize(values):
    result = {"invalid_pearson_count": int((~values["pearson_valid"]).sum())}
    for name in ("ssim", "pearson"):
        scores = values[name]
        valid = name != "pearson" or values["pearson_valid"].all()
        result[name] = float(scores.mean()) if valid else None
        result[name + "_per_plane"] = [float(scores[:, j].mean()) if name != "pearson" or values["pearson_valid"][:, j].all() else None for j in range(4)]
    return result


def metric_summary(result):
    keys = ("ssim", "pearson", "ssim_per_plane", "pearson_per_plane", "invalid_pearson_count")
    return {section: {key: value[key] for key in keys} for section, value in {"aggregate": result["aggregate"], **result["groups"]}.items()}


def write_note(path, result):
    common = {"aggregate": result["reference"]["aggregate"], **result["reference"]["groups"]}
    r49, r56 = result["saved_comparisons"]["R49"], result["saved_comparisons"]["R56"]
    rows = ["| Group | Common SSIM | R49 SSIM | R56 SSIM | Common Pearson | R49 Pearson | R56 Pearson |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in ["aggregate"] + NAMES + ["mixed"]:
        values = [common[group]["ssim"], r49["metrics"][group]["ssim"], r56["metrics"][group]["ssim"], common[group]["pearson"], r49["metrics"][group]["pearson"], r56["metrics"][group]["pearson"]]
        rows.append("| " + group + " | " + " | ".join(f"{value:.6f}" if value is not None else "undefined" for value in values) + " |")
    text = f"""# Common-image diagnostic

This is an abstract metric reference, **not a DOE, a physically valid baseline,
a submission candidate, or a quality upper bound**. It produces no phase mask
and uses no propagation, neural inference, optimization, or official test data.

For each of the previously used 256 selection quartets: independently peak-normalize
the four targets, take their arithmetic mean, peak-normalize that mean, save/read
an actual uint8 grayscale PNG, then use that identical image in all four planes.
Targets and metrics use the existing v2 protocol. Source IDs and manifest SHA256
are embedded in `common_image_diagnostic.json`; all paths are repository-relative.

R49 is its saved best-selection result at additional step {r49['step']}.
R56 is the latest completed validation row available at this diagnostic snapshot,
step {r56['step']}; it is not claimed to be the final or selected checkpoint.
No network was re-evaluated. Historical model metrics were computed on GPU;
the common reference used CPU Torch {result['runtime']['torch']}. Small numerical
backend differences are possible, so this is not a strict paired significance test.

{chr(10).join(rows)}

The identical reconstruction has exactly zero mean diagonal-minus-off-diagonal
depth margin, both for SSIM and Pearson: every reconstruction row has the same
four target scores. Finite-precision maximum absolute observed margins are
{result['reference']['max_abs_depth_margin']['ssim']:.3g} / {result['reference']['max_abs_depth_margin']['pearson']:.3g}.

The common reference shows how much aggregate similarity can exist with no depth
separation. It does not quantify which fraction of a model's gain came from
separation, and surpassing or falling below it does not establish physical
feasibility or optimality. Assess actual depth separation with the model's
correct-versus-wrong-plane metric matrix and visual reconstructions alongside
the ordinary scores. The current comparison alone cannot justify calling a
higher SSIM a better separation of four objects.

Reproduce this snapshot's comparison row:

```powershell
.venv\\Scripts\\python.exe -m scripts.diagnose_common_image --history-step {r56['step']}
```

The full history-file SHA may change while training continues; the selected
validation row has its own SHA and line number. The target-source manifest is
fixed, and no checkpoint file is opened.
"""
    path.write_text(text, encoding="utf-8")


@torch.no_grad()
def run(cfg):
    started = time.perf_counter()
    torch.set_num_threads(cfg.threads)
    manifest_bytes, manifest_source = read_source(cfg.r49_run / "validation_tuples.json")
    manifest = json.loads(manifest_bytes)
    r56_manifest_bytes, r56_manifest_source = read_source(cfg.r56_run / "validation_tuples.json")
    if json.loads(r56_manifest_bytes) != manifest or any(len(rows) != 64 for rows in manifest.values()) or list(manifest) != NAMES + ["mixed"]:
        raise ValueError("Expected the same original 64-per-group selection manifest")
    r49_bytes, r49_source = read_source(cfg.r49_run / "metrics.json")
    r49 = json.loads(r49_bytes)
    r56, r56_source = latest_validation(cfg.r56_run / "history.jsonl", cfg.history_step)
    if any(value["protocol"] != PEAK_PROTOCOL or value["tuples"] != 256 for value in (r49, r56)):
        raise ValueError("Comparison requires saved 256-quartet v2 scores")
    configs = [json.loads((path / "config.json").read_bytes()) for path in (cfg.r49_run, cfg.r56_run)]
    if configs[0]["data_seed"] != configs[1]["data_seed"]:
        raise ValueError("Comparison split seeds must match")
    bank = ImageBank(seed=configs[0]["data_seed"])
    arrays, margin_max = {}, {"ssim": 0.0, "pearson": 0.0}
    for group, rows in manifest.items():
        records = np.asarray(rows)
        for dataset in range(3):
            if not np.isin(records[..., 1][records[..., 0] == dataset], bank.val[dataset]).all():
                raise ValueError("Manifest contains a source outside the fixed validation split")
        values = {"ssim": [], "pearson": [], "pearson_valid": []}
        for start in range(0, len(records), cfg.batch_size):
            target = prepare_target(bank.fetch(records[start:start + cfg.batch_size], "cpu"), PEAK_PROTOCOL)
            common = common_image(target)
            correlation, valid = pearson(common, target, return_valid=True)
            batch = {"ssim": ssim(common, target), "pearson": correlation, "pearson_valid": valid}
            for name in ("ssim", "pearson"):
                matrix = batch[name][:, None].expand(-1, 4, -1)
                diagonal = matrix.diagonal(dim1=1, dim2=2).sum(1)
                margin = diagonal / 4 - (matrix.sum((1, 2)) - diagonal) / 12
                margin_max[name] = max(margin_max[name], float(margin.abs().max()))
            for name, scores in batch.items():
                values[name].append(scores.numpy())
        arrays[group] = {name: np.concatenate(scores) for name, scores in values.items()}
    pooled = {name: np.concatenate([values[name] for values in arrays.values()]) for name in next(iter(arrays.values()))}
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True, "physical_reconstruction": False, "quality_upper_bound": False,
        "neural_inference_used": False, "optimization_used": False, "official_test_used": False,
        "method": "peak-normalized targets -> arithmetic mean of four -> own peak -> uint8 PNG roundtrip -> identical image in all four planes",
        "protocol": PEAK_PROTOCOL, "preprocessing": PREPROCESS_VERSION,
        "selection": {"source": manifest_source, "r56_manifest_source": r56_manifest_source, "source_namespace": "official_training_files_validation_rows", "split_seed": bank.seed, "source_ids": manifest},
        "reference": {"aggregate": summarize(pooled), "groups": {group: summarize(values) for group, values in arrays.items()}, "max_abs_depth_margin": margin_max},
        "saved_comparisons": {
            "R49": {"source": r49_source, "step": r49["selected_step"], "metrics": metric_summary(r49)},
            "R56": {"source": r56_source, "step": r56_source["step"], "metrics": metric_summary(r56)},
        },
        "runtime": {"device": "cpu", "torch": torch.__version__, "numpy": np.__version__, "pillow": PIL.__version__, "threads": cfg.threads, "batch_size": cfg.batch_size, "elapsed_seconds": time.perf_counter() - started},
        "code_sources": [read_source(Path(path))[1] for path in ("scripts/diagnose_common_image.py", "src/data.py", "src/protocol.py", "src/metrics.py")],
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_note(cfg.output.with_suffix(".md"), result)
    print(json.dumps({"output": cfg.output.as_posix(), "reference": result["reference"]["aggregate"], "r56_step": r56_source["step"], "seconds": result["runtime"]["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-step", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    run(Config(history_step=args.history_step, batch_size=args.batch_size, threads=args.threads))


if __name__ == "__main__":
    main()
