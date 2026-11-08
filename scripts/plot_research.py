"""Plot archived selection-validation results; never loads models or datasets."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np


PROTOCOL = "fresnel_png_v2_peak"
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#7A5C00")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_id(record):
    return record["run"].split("_", 1)[0].upper()


def resolve_run(records, name):
    matches = [record for record in records if name.lower() in (record["run"].lower(), run_id(record).lower())]
    if len(matches) != 1:
        raise ValueError(f"Run must resolve uniquely in the registry: {name}")
    return matches[0]


def valid_score(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def validate_runs(records):
    if len({record["run"] for record in records}) != len(records):
        raise ValueError("Requested runs must be distinct")
    manifests = set()
    for record in records:
        manifest = record["validation_manifest"]
        if record["status"] != "completed" or record.get("diagnostic_only", False):
            raise ValueError(f"Only completed research runs are plotted: {record['run']}")
        if record["protocol"] != PROTOCOL or record["metrics"]["protocol"] != PROTOCOL:
            raise ValueError("Only v2 peak-normalized runs are comparable in these figures")
        if manifest["split"] != "selection_validation" or manifest["test_used"] or record["metrics"].get("test_used", False):
            raise ValueError("These figures are restricted to selection validation")
        manifests.add(manifest["sha256"])
        for metric in ("ssim", "pearson"):
            if not valid_score(record["metrics"]["aggregate"].get(metric)):
                raise ValueError(f"Undefined {metric} for {record['run']}")
        curve = record["validation_curve"]
        steps = [point["step"] for point in curve]
        if not steps or steps != sorted(set(steps)) or steps[0] < 0:
            raise ValueError(f"Expected unique increasing nonnegative updates: {record['run']}")
        if steps[-1] > record["budget"]["completed_step"]:
            raise ValueError("Validation curve exceeds completed update budget")
        for point in curve:
            if point["protocol"] != PROTOCOL:
                raise ValueError("Mixed protocols in a learning curve")
            if not valid_score(point["aggregate"].get("ssim")):
                raise ValueError("Missing or nonfinite SSIM in a learning curve")
            pearson = point["aggregate"].get("pearson")
            if pearson is not None and not valid_score(pearson):
                raise ValueError("Nonfinite Pearson in a learning curve")
    if len(manifests) != 1:
        raise ValueError("Runs use different validation manifests; refusing an unpaired visual comparison")


def style_axis(axis):
    axis.grid(color="#D9DEE3", linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#ABB3BA")
    axis.tick_params(colors="#39434C", labelsize=10)


def add_point_labels(figure, axis, points):
    """Choose short label offsets that avoid already placed labels and markers."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    occupied = []
    pixels = [axis.transData.transform((x, y)) for x, y, _ in points]
    offsets = ((8, 8), (8, -16), (-8, 8), (-8, -16), (0, 18), (0, -24), (18, 0), (-18, 0))
    for x, y, label in points:
        best = None
        best_score = float("inf")
        for dx, dy in offsets:
            text = axis.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                                 ha="right" if dx < 0 else "left" if dx > 0 else "center",
                                 va="center", fontsize=10, fontweight="bold", color="#25313C")
            box = text.get_window_extent(renderer).expanded(1.12, 1.3)
            score = sum(box.overlaps(other) for other in occupied) + sum(box.contains(*point) for point in pixels)
            if not axis.bbox.contains(box.x0, box.y0) or not axis.bbox.contains(box.x1, box.y1):
                score += 10
            if score < best_score:
                if best is not None:
                    best.remove()
                best, best_box, best_score = text, box, score
            else:
                text.remove()
            if score == 0:
                break
        occupied.append(best_box)


def save_figure(figure, directory, name):
    paths = []
    for extension in ("png", "svg"):
        path = directory / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else None
        figure.savefig(path, dpi=190, facecolor="white", metadata=metadata)
        paths.append(path)
    plt.close(figure)
    return paths


def plot_scatter(records, selected, directory):
    figure = plt.figure(figsize=(8.8, 7.0))
    axis = figure.add_axes((0.12, 0.37, 0.82, 0.46))
    style_axis(axis)
    points = []
    for number, record in enumerate(records):
        scores = record["metrics"]["aggregate"]
        x, y = scores["ssim"], scores["pearson"]
        chosen = record["run"] == selected
        axis.scatter(x, y, s=170 if chosen else 83, color=COLORS[number % len(COLORS)],
                     marker="*" if chosen else MARKERS[number % len(MARKERS)],
                     edgecolor="#263238" if chosen else "white", linewidth=0.8, zorder=3)
        points.append((x, y, run_id(record) + (" *" if chosen else "")))
    for coordinate, setter in ((0, axis.set_xlim), (1, axis.set_ylim)):
        values = [point[coordinate] for point in points]
        padding = max(0.02, (max(values) - min(values)) * 0.22)
        setter(min(values) - padding, max(values) + padding)
    axis.set_xlabel("Mean local SSIM  ↑", fontsize=11, labelpad=8)
    axis.set_ylabel("Mean Pearson correlation  ↑", fontsize=11, labelpad=9)
    add_point_labels(figure, axis, points)
    figure.text(0.12, 0.945, "Checkpoint quality: two complementary metrics", fontsize=16, fontweight="bold")
    count = records[0]["validation_manifest"]["total_quartets"]
    figure.text(0.12, 0.900, f"Selection validation · {count:,} matched quartets · v2 · PNG roundtrip", fontsize=10, color="#4D5963")
    rows = []
    for record in records:
        resumed = bool(record["config"].get("resume"))
        rows.append([run_id(record) + (" +" if resumed else ""), f"{record['selected_step']:,}",
                     f"{record['budget']['completed_step']:,}", f"{record['config']['parameters'] / 1000:.1f}"])
    table_axis = figure.add_axes((0.14, 0.075, 0.78, 0.19))
    table_axis.axis("off")
    table = table_axis.table(cellText=rows, colLabels=["Run", "Checkpoint update", "Run updates", "Parameters (k)"],
                            cellLoc="center", loc="center", bbox=(0, 0, 1, 1))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor("#E9EEF2" if row == 0 else "#F5F7F9" if row % 2 else "white")
        if row == 0:
            cell.set_text_props(weight="bold", color="#30404C")
        elif column == 0:
            cell.set_text_props(weight="bold", color=COLORS[(row - 1) % len(COLORS)])
    batches = {record["config"]["batch_size"] for record in records}
    seeds = {record["config"]["seed"] for record in records}
    note = "Selection scores; run-specific budgets are shown above. Seeds and batch sizes are recorded in the figure provenance."
    if len(batches) == 1 and len(seeds) == 1:
        note = f"All runs: batch {next(iter(batches))}, seed {next(iter(seeds))}. These are selection scores, not test scores."
    if any(record["config"].get("resume") for record in records):
        note = "+ Warm start: update counts are additional updates in that run; prior training is excluded."
    figure.text(0.12, 0.025, note, fontsize=8.2, color="#59656E")
    if selected:
        axis.text(0.98, 0.025, "* Externally selected model", transform=axis.transAxes, ha="right", fontsize=8.5, color="#59656E")
    return save_figure(figure, directory, "selection_ssim_pearson")


def plot_learning_curves(records, selected, directory):
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.7))
    handles = []
    for number, record in enumerate(records):
        color, marker = COLORS[number % len(COLORS)], MARKERS[number % len(MARKERS)]
        curve = record["validation_curve"]
        updates = [point["step"] for point in curve]
        resumed = bool(record["config"].get("resume"))
        label = run_id(record) + (" (additional updates)" if resumed else "")
        handles.append(Line2D([], [], color=color, marker=marker, linewidth=2, markersize=5, label=label))
        for axis, metric in zip(axes, ("ssim", "pearson")):
            values = [np.nan if point["aggregate"].get(metric) is None else point["aggregate"][metric] for point in curve]
            axis.plot(updates, values, color=color, marker=marker, markersize=4, linewidth=1.9, alpha=0.95)
            axis.scatter(record["selected_step"], record["metrics"]["aggregate"][metric],
                         s=135 if record["run"] == selected else 85, marker="*" if record["run"] == selected else "o",
                         facecolors="none", edgecolors=color, linewidth=1.4, zorder=4)
    for axis, title, ylabel in zip(axes, ("Local SSIM", "Pearson correlation"), ("Mean SSIM", "Mean Pearson")):
        style_axis(axis)
        axis.set_title(title, fontsize=12, loc="left", pad=13, fontweight="bold")
        axis.set_xlabel("Optimizer updates in this run", fontsize=10, labelpad=8)
        axis.set_ylabel(ylabel, fontsize=10)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axis.set_xlim(left=0)
        axis.margins(y=0.12)
    figure.suptitle("Learning curves on selection validation", fontsize=16, fontweight="bold", x=0.07, ha="left", y=0.97)
    figure.text(0.07, 0.900, "v2 · PNG roundtrip · Logged evaluations only; lines join observations without smoothing", fontsize=9.7, color="#4D5963")
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.065), ncol=min(5, len(handles)), frameon=False, fontsize=9)
    resumed = any(record["config"].get("resume") for record in records)
    note = "Hollow circles: saved checkpoint. Each run uses its own update counter; no cumulative offset is inferred."
    if resumed:
        note = "Hollow circles: saved checkpoint. Resumed curves show additional updates, excluding all prior training."
    if selected:
        note += " Star: externally selected."
    figure.text(0.07, 0.025, note, fontsize=8.4, color="#59656E")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.24, top=0.79, wspace=0.26)
    return save_figure(figure, directory, "selection_learning_curves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("experiments/run_registry.json"))
    parser.add_argument("--runs", nargs="+", required=True, help="Exact run names or IDs such as R31")
    parser.add_argument("--selected", help="Optional externally selected run, already included in --runs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/research_figures_smoke"))
    args = parser.parse_args()
    registry_bytes = args.registry.read_bytes()
    registry = json.loads(registry_bytes.decode("utf-8-sig"))
    records = [resolve_run(registry["runs"], name) for name in args.runs]
    validate_runs(records)
    selected = resolve_run(records, args.selected)["run"] if args.selected else None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "neural-holography-research-v1"})
    paths = plot_scatter(records, selected, args.output_dir)
    paths += plot_learning_curves(records, selected, args.output_dir)
    provenance = {
        "registry": {"name": args.registry.name, "sha256": hashlib.sha256(registry_bytes).hexdigest()},
        "script": {"path": "scripts/plot_research.py", "sha256": file_hash(__file__)},
        "scope": "Selection validation after phase and reconstruction PNG roundtrips; not held-out test",
        "protocol": PROTOCOL, "validation_manifest_sha256": records[0]["validation_manifest"]["sha256"],
        "selected_run": selected, "selection_source": "Explicit CLI flag only; plotting does not select a winner",
        "update_axis": "Actual local optimizer updates from each validation_curve. Resumes use additional updates in their own run; no cumulative offset or earlier budget is inferred.",
        "runs": [{"run": record["run"], "config": record["config"], "selected_step": record["selected_step"],
                  "budget": record["budget"], "plotted_scores": {metric: record["metrics"]["aggregate"][metric] for metric in ("ssim", "pearson")},
                  "plotted_curve": [{"step": point["step"], **{metric: point["aggregate"].get(metric) for metric in ("ssim", "pearson")}} for point in record["validation_curve"]]} for record in records],
        "artifacts": [{"name": path.name, "sha256": file_hash(path)} for path in paths],
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
    }
    (args.output_dir / "figure_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": args.output_dir.as_posix(), "runs": [record["run"] for record in records], "figures": [path.name for path in paths]}))


if __name__ == "__main__":
    main()
