"""Generate reproducible EDA figures for DeepWeeds fold 0."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "labels"
IMAGES = ROOT / "data" / "images"
OUT = ROOT / "eda"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    labels = pd.read_csv(LABELS / "labels.csv").drop_duplicates("Label").sort_values("Label")
    names = labels.set_index("Label")["Species"].to_dict()
    frames = {
        split: pd.read_csv(LABELS / f"{split}_subset0.csv")
        for split in ("train", "val", "test")
    }
    counts = pd.concat(
        [frame.assign(split=split).groupby(["split", "Label"]).size().rename("n")
         for split, frame in frames.items()]
    ).reset_index()
    counts["class"] = counts["Label"].map(names)
    pivot = counts.pivot(index="class", columns="split", values="n").reindex(labels["Species"])
    axis = pivot.plot(kind="bar", figsize=(14, 5), color=["#4C78A8", "#F58518", "#54A24B"])
    axis.set(title="DeepWeeds fold 0 — class distribution", xlabel="Class", ylabel="Image count")
    axis.legend(title="Split")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "class_distribution_fold0.png", dpi=180)
    plt.close()

    samples = frames["train"].groupby("Label", group_keys=False).sample(n=2, random_state=7)
    figure, axes = plt.subplots(9, 2, figsize=(7, 27))
    for label in range(9):
        rows = samples[samples.Label.eq(label)]
        for col, (_, row) in enumerate(rows.iterrows()):
            with Image.open(IMAGES / row.Filename) as image:
                axes[label, col].imshow(image.convert("RGB"))
            axes[label, col].set_title(names[label])
            axes[label, col].axis("off")
    figure.suptitle("DeepWeeds fold 0 — training examples", y=1.001, fontsize=14)
    figure.tight_layout()
    figure.savefig(OUT / "sample_images_fold0.png", dpi=160, bbox_inches="tight")
    plt.close(figure)

    summary = pivot.copy()
    summary["total"] = summary.sum(axis=1)
    summary.to_csv(OUT / "class_counts_fold0.csv")
    print(f"Wrote EDA figures and counts to {OUT}")


if __name__ == "__main__":
    main()
