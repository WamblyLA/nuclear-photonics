import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path):
    with Image.open(path) as image:
        if image.mode != "L" or image.size != (128, 128):
            raise ValueError(f"Expected a 128 x 128 grayscale PNG: {path}")
        return np.array(image, dtype=np.uint8)


def load_examples(runs, group):
    examples = []
    reference = None
    for run in runs:
        directory = run / "examples" / group
        targets = [load_image(directory / f"target_{plane}.png") for plane in range(1, 5)]
        if reference is None:
            reference = targets
        for plane, (expected, target) in enumerate(zip(reference, targets), start=1):
            if not np.array_equal(expected, target):
                raise ValueError(f"Target mismatch in {run}, {group}, plane {plane}; refusing an unaligned comparison")
        reconstructed = [load_image(directory / f"reconstruction_{plane}.png") for plane in range(1, 5)]
        phase = load_image(directory / "phase.png")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        examples.append((run, config, reconstructed, phase))
    return reference, examples


def plot_examples(runs, group, output):
    targets, examples = load_examples(runs, group)
    figure, axes = plt.subplots(len(examples) + 1, 5, figsize=(12, 2.1 * (len(examples) + 1)), squeeze=False)
    titles = [f"Plane {plane}: {distance} mm" for plane, distance in enumerate((24.1, 25.3, 26.5, 27.7), start=1)]
    titles.append("Phase mask (8-bit codes)")
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=11, pad=9)
    for column, target in enumerate(targets):
        axes[0, column].imshow(target, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axes[0, 4].text(0.5, 0.5, "No target phase", ha="center", va="center", color="0.4", transform=axes[0, 4].transAxes)
    axes[0, 0].set_ylabel("Targets", fontsize=11, rotation=0, ha="right", va="center", labelpad=12)
    for row, (run, config, reconstructed, phase) in enumerate(examples, start=1):
        for column, array in enumerate(reconstructed + [phase]):
            axes[row, column].imshow(array, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        label = f"{run.name.split('_')[0]}: {config['model']}\n{config.get('phase_head', 'tanh')} head\nSSIM weight {config.get('ssim_weight', 0.1):g}"
        if config.get("rank_weight", 0):
            label += f"\nRank weight {config['rank_weight']:g}"
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=0, ha="right", va="center", labelpad=12)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.suptitle(f"{group}: the same saved validation targets across models", fontsize=14, y=0.985)
    figure.text(0.5, 0.012, "Fixed grayscale range 0-255 for all images. Phase codes: 0 = -pi, 255 = +pi. One example, not aggregate quality.", ha="center", fontsize=9)
    figure.subplots_adjust(left=0.17, right=0.99, top=0.93, bottom=0.055, wspace=0.05, hspace=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)
    print(json.dumps({"output": str(output), "group": group, "runs": [str(run) for run in runs], "targets_identical": True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--group", default="MNIST")
    parser.add_argument("--output", type=Path, default=Path("outputs/cycle2_reconstructions.png"))
    args = parser.parse_args()
    runs = [run if run.exists() else Path("outputs") / run for run in args.runs]
    plot_examples(runs, args.group, args.output)


if __name__ == "__main__":
    main()
