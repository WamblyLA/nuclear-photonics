"""Plot existing moment-bound summaries; no inference or dataset access."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


GROUPS = ("CIFAR10", "MNIST", "FashionMNIST", "mixed")
COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7")


def load_groups(source):
    if not source.get("diagnostic_only") or source["provenance"]["test_used"] or source["provenance"].get("reserve_used", False):
        raise ValueError("Expected a diagnostic based on existing training/selection data only")
    rows = []
    for name in GROUPS:
        group = source["groups"]["selection_" + name]
        count = group["quartets"]
        mean = group["mean_max_pair_intensity_l1_error_sum_lower_bound"]
        violations = group["quartets_with_any_violation"]
        if count < 1 or not 0 <= mean <= 2 or not 0 <= violations <= count or group["zero_target_images"]:
            raise ValueError(f"Invalid or incomplete moment diagnostic for {name}")
        if abs(group["fraction_with_any_violation"] - violations / count) > 1e-12:
            raise ValueError(f"Inconsistent violation count for {name}")
        rows.append({"group": name, "quartets": count, "violations": violations, "mean_strongest_pair_lower_bound": mean})
    return rows


def plot(source, rows, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "moment-compatibility-v1"})
    figure = plt.figure(figsize=(10.4, 6.6))
    axis = figure.add_axes((0.19, 0.32, 0.53, 0.45))
    values = [row["mean_strongest_pair_lower_bound"] for row in rows]
    positions = range(len(rows))
    axis.barh(positions, values, color=COLORS, height=0.55, edgecolor="white", linewidth=0.6)
    axis.set_yticks(list(positions), [row["group"] for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.2)
    axis.xaxis.set_major_locator(MultipleLocator(0.1))
    axis.set_xlabel("Mean strongest-pair normalized-intensity L₁ lower bound", fontsize=10.3, labelpad=11)
    axis.grid(axis="x", color="#DFE5EA", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#ABB3BA")
    axis.tick_params(axis="y", length=0, pad=12, labelsize=11)
    axis.tick_params(axis="x", colors="#4C5963", labelsize=10)
    for index, row in enumerate(rows):
        value = row["mean_strongest_pair_lower_bound"]
        axis.text(value + 0.012, index, f"{value:.3f}", va="center", fontsize=11, fontweight="bold", color="#27343F")
        axis.text(1.18, index, f"{row['violations']} / {row['quartets']}  ({row['violations'] / row['quartets']:.0%})",
                  va="center", ha="center", transform=axis.get_yaxis_transform(), fontsize=11, color="#27343F")
    axis.text(1.18, 1.09, "Exact compatibility\nviolations", transform=axis.transAxes, va="bottom", ha="center", fontsize=10, fontweight="bold", color="#35444F")
    figure.text(0.075, 0.94, "Target intensities violate exact compatibility", fontsize=17, fontweight="bold", color="#182630")
    figure.text(0.075, 0.885, "Selection targets · strongest tested plane-pair / frequency constraint per quartet", fontsize=10.6, color="#4C5963")
    figure.text(0.075, 0.195, r"For the violating pair:  $\Vert q_i-p_i\Vert_1 + \Vert q_j-p_j\Vert_1 \;\geq\;$ moment violation", fontsize=11, color="#27343F")
    figure.text(0.075, 0.145, "q = target intensity / total target intensity; p = any realizable normalized intensity.", fontsize=9.5, color="#52616C")
    figure.text(0.075, 0.103, "Bars are recorded group means. Per-quartet distributions were not saved; no uncertainty interval is shown.", fontsize=8.9, color="#52616C")
    figure.text(0.075, 0.035, "NOT a bound on SSIM, Pearson or amplitude MSE", fontsize=11, fontweight="bold", color="#5C4100",
                bbox={"boxstyle": "round,pad=0.55", "facecolor": "#FFF1CC", "edgecolor": "none"})
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for extension in ("png", "svg"):
        path = output_dir / f"moment_compatibility.{extension}"
        figure.savefig(path, dpi=190, facecolor="white", metadata={"Date": None} if extension == "svg" else None)
        files.append(path)
    plt.close(figure)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/moment_compatibility_diagnostic.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/research"))
    args = parser.parse_args()
    payload = args.input.read_bytes()
    source = json.loads(payload.decode("utf-8-sig"))
    rows = load_groups(source)
    files = plot(source, rows, args.output_dir)
    provenance = {
        "input": {"name": args.input.name, "sha256": hashlib.sha256(payload).hexdigest()},
        "script": {"path": "scripts/plot_moment_compatibility.py", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "selection_only": True, "test_used": False, "new_inference": False,
        "plotted_statistic": "Mean over quartets of the maximum positive target-moment violation over all tested plane pairs and frequencies",
        "distribution_available": False, "confidence_intervals": False,
        "not_a_bound_on": source["not_a_bound_on"], "bound": source["bound"], "l1_bound": source["l1_bound"],
        "source_provenance": source["provenance"], "plotted_groups": rows,
        "artifacts": [{"name": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
        "matplotlib": matplotlib.__version__,
    }
    (args.output_dir / "moment_compatibility.provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"figures": [path.as_posix() for path in files], "groups": rows}))


if __name__ == "__main__":
    main()
