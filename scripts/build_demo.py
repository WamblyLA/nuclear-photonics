"""Build an offline, reproducible five-set gallery from a portable checkpoint."""

import argparse
import html
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import file_hash, predict, read_input
from src.protocol import PEAK_PROTOCOL, prepare_target


BACKGROUND = "#f5f7fa"
INK = "#15283e"
MUTED = "#536579"
ACCENT = "#086f83"
GRID = "#ccd5df"


@dataclass
class Config:
    checkpoint: Path
    examples: Path = Path("examples")
    output: Path = Path("docs/demo")
    smoke: bool = False


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def number(value):
    return "undefined" if value is None else f"{value:.4f}"


def font(size):
    # Pillow's bundled font makes rendering independent of installed system fonts.
    return ImageFont.load_default(size=size)


def read_examples(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["split"] != "selection_validation" or manifest["test_used"] is not False:
        raise ValueError("Demonstrations must be the fixed selection-validation examples")
    if [item["set"] for item in manifest["sets"]] != [1, 2, 3, 4, 5]:
        raise ValueError("Expected all five fixed example sets in their original order")
    for record in manifest["sets"]:
        if len(record["inputs"]) != 4:
            raise ValueError("Each example must contain four inputs")
        objects = {(item["dataset"], item["class_label"]) for item in record["inputs"]}
        if len(objects) != 4:
            raise ValueError("Each quartet must contain four distinct dataset/class pairs")
        for plane, item in enumerate(record["inputs"], start=1):
            expected = f"input_set{record['set']}/image{plane}.png"
            if item["path"] != expected:
                raise ValueError("Unexpected example path or ordering")
            path = directory / expected
            read_input(path)
            if file_hash(path) != item["sha256"]:
                raise ValueError(f"Example checksum mismatch: {expected}")
    return manifest, file_hash(manifest_path)


def paste_image(canvas, array, x, y, size=256):
    preview = Image.fromarray(array).resize((size, size), Image.Resampling.NEAREST).convert("RGB")
    canvas.paste(preview, (x, y))
    ImageDraw.Draw(canvas).rectangle((x - 1, y - 1, x + size, y + size), outline=GRID)


def contact_sheet(folder, record, metrics, smoke=False):
    folder = Path(folder)
    originals = np.stack([read_input(folder / f"target_{index}.png") for index in range(1, 5)])
    values = torch.from_numpy(originals).float()[None] / 255
    target_display = (prepare_target(values, metrics["protocol"]) * 255).round().to(torch.uint8)[0].numpy()
    changed = bool(np.any(target_display != originals))
    rows = [("INPUT PNG", "original uint8", originals)]
    if changed:
        rows.append(("TARGET", "per-image max = 1", target_display))
        for index, array in enumerate(target_display, start=1):
            Image.fromarray(array).save(folder / f"target_display_{index}.png")
    else:
        rows[0] = ("INPUT / TARGET", "identical display", originals)
    reconstructed = np.stack([read_input(folder / f"reconstruction_{index}.png") for index in range(1, 5)])
    rows.append(("RECONSTRUCTION", "saved uint8 PNG", reconstructed))
    width, height = 1360, 200 + 326 * len(rows) + 70
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 9), fill=ACCENT)
    draw.text((32, 30), f"SET {record['set']:02d}  |  FOUR-DEPTH RECONSTRUCTION", font=font(31), fill=INK)
    draw.text((32, 77), f"Local SSIM {number(metrics['mean']['ssim'])}    Pearson {number(metrics['mean']['pearson'])}    MSE {number(metrics['mean']['mse'])}", font=font(22), fill=ACCENT)
    subtitle = "Intermediate checkpoint: pipeline smoke test, not the final quality result." if smoke else f"Saved-phase Fresnel amplitude | protocol: {metrics['protocol']}"
    draw.text((32, 112), subtitle, font=font(19), fill=MUTED)
    x_positions = [206 + index * 279 for index in range(4)]
    for index, (x, item, scores) in enumerate(zip(x_positions, record["inputs"], metrics["per_plane"])):
        draw.text((x, 150), f"{scores['distance_m'] * 1000:.1f} mm", font=font(23), fill=INK)
        draw.text((x, 179), f"{item['dataset']} / class {item['class_label']}", font=font(17), fill=MUTED)
    for row_index, (title, description, arrays) in enumerate(rows):
        y = 213 + row_index * 326
        draw.text((28, y + 7), title, font=font(16), fill=INK)
        draw.text((28, y + 34), description, font=font(15), fill=MUTED)
        for x, array in zip(x_positions, arrays):
            paste_image(canvas, array, x, y)
        if row_index == len(rows) - 1:
            for x, scores in zip(x_positions, metrics["per_plane"]):
                draw.text((x, y + 272), f"SSIM {number(scores['ssim'])}   P {number(scores['pearson'])}", font=font(19), fill=ACCENT)
    footer = "Targets: float32 per-image peak normalization; target previews are rounded for display." if metrics["protocol"] == PEAK_PROTOCOL else "Targets: original PNG / 255. Reconstruction display follows the saved checkpoint protocol."
    draw.text((32, height - 55), footer, font=font(17), fill=MUTED)
    draw.text((32, height - 30), "All image previews use nearest-neighbor scaling. No contrast enhancement or denoising is applied.", font=font(17), fill=MUTED)
    canvas.save(folder / "contact_sheet.png")
    return changed


def phase_sheet(folder, record):
    folder = Path(folder)
    phase = read_input(folder / "phase.png")
    canvas = Image.new("RGB", (870, 390), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 23), f"SET {record['set']:02d}  |  ONE PHASE MASK", font=font(28), fill=INK)
    paste_image(canvas, phase, 28, 82)
    for line, text in enumerate(("128 x 128 phase-only DOE", "One network forward pass", "No per-input phase optimization", "Saved PNG is decoded before Fresnel propagation")):
        draw.text((320, 92 + line * 38), text, font=font(19), fill=INK if line == 0 else MUTED)
    ramp = np.tile(np.arange(256, dtype=np.uint8), (20, 1))
    canvas.paste(Image.fromarray(ramp).resize((450, 23), Image.Resampling.NEAREST).convert("RGB"), (320, 268))
    for x, text in ((320, "-pi / code 0"), (510, "0 rad"), (665, "+pi / code 255")):
        draw.text((x, 302), text, font=font(17), fill=MUTED)
    draw.text((28, 357), "Codes 0 and 255 represent equivalent phase endpoints; the mask is shown without filtering.", font=font(17), fill=MUTED)
    canvas.save(folder / "phase_panel.png")


def gallery_html(summary):
    escape = html.escape
    cards = []
    for record in summary["sets"]:
        folder = record["directory"]
        rows = "".join(
            f"<tr><td>{item['plane']}</td><td>{item['distance_m'] * 1000:.1f} mm</td><td>{number(item['ssim'])}</td><td>{number(item['pearson'])}</td><td>{number(item['mse'])}</td></tr>"
            for item in record["per_plane"]
        )
        originals = " · ".join(f'<a href="{folder}/target_{plane}.png">input {plane}</a>' for plane in range(1, 5))
        reconstructions = " · ".join(f'<a href="{folder}/reconstruction_{plane}.png">plane {plane}</a>' for plane in range(1, 5))
        target_note = "The first row preserves original PNG values; the second shows peak-normalized targets." if record["normalized_target_display_differs"] else "Original input and evaluated target have the same visible uint8 display."
        cards.append(f'''<section class="set" id="set{record['set']}">
<div class="set-heading"><h2>Set {record['set']:02d}</h2><p>Mean local SSIM <strong>{number(record['mean']['ssim'])}</strong> · Pearson <strong>{number(record['mean']['pearson'])}</strong></p></div>
<a href="{folder}/contact_sheet.png"><img class="sheet" src="{folder}/contact_sheet.png" alt="Set {record['set']}: original targets, evaluated target display, and reconstructions at four depths" loading="lazy"></a>
<p class="caption">{target_note} Reconstruction previews use the actual saved PNGs. Metrics compare those reconstructions with float32 targets, without rounding the normalized target for evaluation.</p>
<div class="detail"><a href="{folder}/phase.png"><img class="phase" src="{folder}/phase_panel.png" alt="Phase mask and encoding for set {record['set']}" loading="lazy"></a>
<div><table><thead><tr><th>Plane</th><th>Depth</th><th>SSIM</th><th>Pearson</th><th>MSE</th></tr></thead><tbody>{rows}</tbody></table>
<p class="downloads">Original PNGs: {originals}<br>Reconstructions: {reconstructions}<br><a href="{folder}/phase.png">phase mask</a> · <a href="{folder}/metrics.json">all metrics + hashes</a></p></div></div></section>''')
    warning = '<aside class="smoke">PIPELINE SMOKE TEST · This intermediate checkpoint verifies reproducibility and presentation. Its reconstruction quality is not a final solution claim.</aside>' if summary["smoke_test"] else ""
    mean = summary["mean"]
    links = "".join(f'<a href="#set{record["set"]}">Set {record["set"]:02d}</a>' for record in summary["sets"])
    normalization = "Each target and reconstructed amplitude is normalized by its own maximum; zero images stay zero." if summary["protocol"] == PEAK_PROTOCOL else "Historical fixed-scale protocol: targets remain PNG / 255, and reconstructed amplitudes are clipped to [0, 1]."
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neural holography | Five fixed quartets</title>
<style>
:root{{--ink:#15283e;--muted:#536579;--accent:#086f83;--line:#dce4ec;--paper:#f5f7fa}}*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,sans-serif}}main{{max-width:1440px;margin:auto;padding:46px 30px 60px}}header{{max-width:1050px}}.eyebrow{{color:var(--accent);font-weight:750;letter-spacing:.14em;font-size:12px}}h1{{font-size:clamp(34px,5vw,60px);line-height:1.08;letter-spacing:-.04em;margin:14px 0 22px}}h2{{font-size:26px;margin:0}}p{{margin:10px 0}}.intro{{font-size:20px;color:var(--muted)}}a{{color:var(--accent);text-underline-offset:3px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:30px 0}}.stat{{background:white;border:1px solid var(--line);border-radius:13px;padding:20px}}.stat strong{{display:block;font-size:31px;line-height:1.2}}.stat span{{font-size:13px;color:var(--muted)}}nav{{display:flex;gap:10px;flex-wrap:wrap;margin:22px 0 30px}}nav a{{background:white;border:1px solid var(--line);border-radius:30px;padding:6px 17px;text-decoration:none}}.provenance{{background:#e9f1f4;border-left:4px solid var(--accent);padding:20px 24px;border-radius:0 12px 12px 0}}.provenance h2{{font-size:20px}}.provenance p,.caption{{color:var(--muted);font-size:14px}}code{{font-size:12px;overflow-wrap:anywhere}}.set{{background:white;border:1px solid var(--line);border-radius:16px;padding:22px;margin-top:28px;overflow:hidden}}.set-heading{{display:flex;gap:20px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin:0 3px 16px}}.sheet{{display:block;width:100%;height:auto;border-radius:8px}}.caption{{margin:14px 3px}}.detail{{display:grid;grid-template-columns:1.2fr 1fr;gap:25px;align-items:center;margin-top:20px}}.phase{{width:100%;display:block;border-radius:8px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}th,td{{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--muted);font-size:12px}}.downloads{{font-size:13px;margin-top:16px}}.smoke{{padding:16px 20px;background:#fff0cf;border:1px solid #e6c47e;border-radius:10px;margin:24px 0}}footer{{font-size:13px;color:var(--muted);margin-top:30px}}@media(max-width:760px){{main{{padding:25px 12px}}.stats{{grid-template-columns:repeat(2,1fr)}}.detail{{grid-template-columns:1fr}}.set{{padding:12px}}.set-heading p{{font-size:14px}}}}@media print{{nav{{display:none}}.set{{break-inside:avoid}}body{{background:white}}}}
</style></head><body><main><header><div class="eyebrow">PHYSICS-INFORMED NEURAL PHASE SYNTHESIS</div><h1>Four targets.<br>One phase mask.</h1><p class="intro">Five fixed quartets, reconstructed at four depths from one neural prediction per quartet. Every phase mask and reconstruction is available as the original 128 × 128 grayscale PNG.</p></header>
{warning}<div class="stats"><div class="stat"><strong>{number(mean['ssim'])}</strong><span>mean local SSIM · 20 planes</span></div><div class="stat"><strong>{number(mean['pearson'])}</strong><span>mean Pearson · 20 planes</span></div><div class="stat"><strong>5 / 20</strong><span>fixed quartets / input images</span></div><div class="stat"><strong>532 nm</strong><span>wavelength · 10 µm source pitch</span></div></div>
<section class="provenance"><h2>Fixed examples, explicit provenance</h2><p>{escape(summary['selection_rule'])}. These examples belong to selection validation, which informed model development. Their scores are descriptive demonstrations, not an independent held-out test estimate. Every predefined set is shown in order.</p><p>{normalization} Reconstruction uses amplitude |E|, not intensity. The saved 8-bit phase PNG is decoded before Fresnel propagation, and reconstruction PNGs are read back before scoring. Local SSIM uses an 11 × 11 Gaussian window, sigma 1.5, valid crop. Undefined Pearson values remain explicit.</p><p>Source manifest SHA-256: <code>{summary['source_manifest_sha256']}</code><br>Checkpoint SHA-256: <code>{summary['checkpoint_sha256']}</code><br>Model: <code>{escape(summary['model'])}</code> · selected step {summary['checkpoint_step']} · protocol <code>{escape(summary['protocol'])}</code></p><p><a href="provenance.json">input provenance and hashes</a> · <a href="summary.json">aggregate metrics and environment</a></p></section>
<nav aria-label="Example sets">{links}</nav>{''.join(cards)}
<footer>Offline gallery generated with <code>python -m scripts.build_demo</code>. All figures and links are local; no remote assets, network inference, contrast enhancement, or per-input phase optimization. Aggregate values weight all four planes in all five sets equally.</footer></main></body></html>'''


def build_demo(config):
    checkpoint, examples, output = Path(config.checkpoint), Path(config.examples), Path(config.output)
    if output.exists():
        raise FileExistsError(f"Demo output already exists: {output}")
    manifest, manifest_sha = read_examples(examples)
    # The gallery accepts portable tensor-only artifacts, not arbitrary pickle checkpoints.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format_version") != 1 or "optimizer" in saved:
        raise ValueError("Use scripts.export_solution to create a portable checkpoint first")
    if saved["config"].get("fixed_train_quartets", 0) > 0:
        raise ValueError("A memorization diagnostic is not a submission demonstration")
    checkpoint_sha = file_hash(checkpoint)
    summary = {
        "smoke_test": bool(config.smoke),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": saved["step"],
        "model": saved["config"]["model"],
        "protocol": saved["config"]["protocol"],
        "selection_rule": manifest["selection_rule"],
        "split": manifest["split"],
        "test_used": False,
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "examples_manifest_sha256": manifest_sha,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__, "pillow": pillow_version, "device": "cpu"},
        "timing_scope": "CPU predict call, including model load, inference, Fresnel propagation, PNG I/O and metrics; excludes figures",
        "sets": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "provenance.json", manifest)
    for record in manifest["sets"]:
        folder = output / f"input_set{record['set']}"
        paths = [examples / item["path"] for item in record["inputs"]]
        started = time.perf_counter()
        metrics = predict(checkpoint, paths, folder, device="cpu")
        elapsed = time.perf_counter() - started
        if metrics["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("Checkpoint changed during gallery generation")
        # Keep published paths local even when callers pass absolute input/checkpoint paths.
        metrics["checkpoint"] = checkpoint.name
        for item in metrics["inputs"]:
            item["path"] = Path(item["path"]).name
        write_json(folder / "metrics.json", metrics)
        changed = contact_sheet(folder, record, metrics, config.smoke)
        phase_sheet(folder, record)
        summary["sets"].append({
            "set": record["set"], "directory": folder.name, "mean": metrics["mean"],
            "per_plane": metrics["per_plane"], "invalid_pearson_count": metrics["invalid_pearson_count"],
            "cpu_predict_seconds": elapsed, "normalized_target_display_differs": changed,
        })
    planes = [plane for record in summary["sets"] for plane in record["per_plane"]]
    summary["mean"] = {
        name: sum(plane[name] for plane in planes) / len(planes) if all(plane[name] is not None for plane in planes) else None
        for name in ("ssim", "pearson", "global_ssim", "mse")
    }
    summary["invalid_pearson_count"] = sum(record["invalid_pearson_count"] for record in summary["sets"])
    if any(value is not None and not math.isfinite(value) for value in summary["mean"].values()):
        raise ValueError("Nonfinite demo metric")
    write_json(output / "summary.json", summary)
    (output / "index.html").write_text(gallery_html(summary), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/solution.pt"))
    parser.add_argument("--examples", type=Path, default=Path("examples"))
    parser.add_argument("--output", type=Path, default=Path("docs/demo"))
    parser.add_argument("--smoke", action="store_true", help="Mark an intermediate checkpoint as a pipeline smoke test")
    args = parser.parse_args()
    result = build_demo(Config(**vars(args)))
    print(json.dumps({"output": str(args.output), "sets": len(result["sets"]), "mean": result["mean"], "smoke_test": result["smoke_test"]}, allow_nan=False))


if __name__ == "__main__":
    main()
