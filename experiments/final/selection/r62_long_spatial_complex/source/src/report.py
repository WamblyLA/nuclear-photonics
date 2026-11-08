import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--plot-runs", nargs="*")
    parser.add_argument("--protocol")
    args = parser.parse_args()
    root = Path("outputs")
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for directory in sorted(root.glob("r*")):
        metrics_path = directory / args.tag / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if args.protocol and metrics.get("protocol") != args.protocol:
            continue
        training_metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("diagnostic_only", False) or config.get("fixed_train_quartets", 0) > 0:
            continue
        row = {"run": directory.name, "scope": args.tag or "selection_validation", "tuples": metrics["tuples"], "selected_step": metrics.get("selected_step", config["steps"]), "model": config["model"], "position": config.get("position", "none"), "seed": config["seed"], "steps": config["steps"], "parameters": config["parameters"], "ssim": metrics["aggregate"]["ssim"], "pearson": metrics["aggregate"]["pearson"], "global_ssim": metrics["aggregate"]["global_ssim"], "mse": metrics["aggregate"]["mse"], "seconds": training_metrics["training_seconds"], "inference_ms": training_metrics["inference_ms_batch1"]}
        row.update({"protocol": metrics.get("protocol", "fresnel_png_v1"), "model_input_protocol": config.get("protocol", "fresnel_png_v1"), "phase_head": config.get("phase_head", "tanh"), "ssim_weight": config.get("ssim_weight", 0.1), "contrast_weight": config.get("contrast_weight", 0.0), "rank_weight": config.get("rank_weight", 0.0), "synthetic_steps": config.get("synthetic_steps", 0)})
        for name, group in metrics["groups"].items():
            row[name + "_ssim"] = group["ssim"]
        rows.append(row)
        if args.plot_runs is not None and directory.name not in args.plot_runs:
            continue
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        validation = [entry for entry in history if "validation" in entry]
        for axis, key in zip(axes, ("ssim", "pearson")):
            axis.plot([entry["step"] for entry in validation], [entry["validation"]["aggregate"][key] for entry in validation], marker=".", label=directory.name)
            axis.set(xlabel="Training updates", ylabel=key.upper(), title=f"Validation {key.upper()} after PNG")
            axis.grid(alpha=0.2)
    if rows:
        prefix = args.tag or args.protocol or ""
        filename = (prefix + "_" if prefix else "") + "comparison.csv"
        for parent in (root, Path("experiments")):
            parent.mkdir(exist_ok=True)
            with (parent / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    if not args.tag:
        fig.savefig(root / "learning_curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
