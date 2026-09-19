import json
import os

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Paths
# ============================================================

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Helper
# ============================================================

def save_figure(filename):
    path = os.path.join(FIGURES_DIR, filename)

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# 1. Linear probe comparison
# ============================================================

linear_probe_path = os.path.join(
    RESULTS_DIR,
    "linear_probe_food101.json",
)

with open(linear_probe_path) as f:
    linear_probe = json.load(f)


random_acc = linear_probe[
    "random_frozen_resnet50_accuracy"
]

simclr_acc = linear_probe[
    "simclr_frozen_resnet50_accuracy"
]


plt.figure(figsize=(7, 5))

labels = [
    "Random\nResNet-50",
    "SimCLR\nResNet-50",
]

values = [
    random_acc,
    simclr_acc,
]

bars = plt.bar(
    labels,
    values,
)

plt.ylabel("Top-1 accuracy (%)")

plt.title(
    "Linear Probe: Random vs SimCLR Representations"
)

plt.ylim(0, 65)

plt.bar_label(
    bars,
    fmt="%.2f%%",
    padding=3,
)

save_figure(
    "linear_probe_comparison.png"
)


# ============================================================
# 2. Fine-tuning comparison
# ============================================================

finetuning_path = os.path.join(
    RESULTS_DIR,
    "food101_finetuning_comparison.json",
)

with open(finetuning_path) as f:
    finetuning = json.load(f)


scratch_acc = (
    finetuning[
        "supervised_from_scratch"
    ]["test_top1"]
)

simclr_ft_acc = (
    finetuning[
        "simclr_pretrained_finetuned"
    ]["test_top1"]
)


plt.figure(figsize=(7, 5))

labels = [
    "Supervised\nfrom scratch",
    "SimCLR pretrained\n+ fine-tuned",
]

values = [
    scratch_acc,
    simclr_ft_acc,
]

bars = plt.bar(
    labels,
    values,
)

plt.ylabel("Top-1 accuracy (%)")

plt.title(
    "Food-101 Downstream Fine-Tuning"
)

plt.ylim(0, 85)

plt.bar_label(
    bars,
    fmt="%.2f%%",
    padding=3,
)

save_figure(
    "finetuning_comparison.png"
)


# ============================================================
# 3. Pretraining duration
# ============================================================

duration_path = os.path.join(
    RESULTS_DIR,
    "training_duration_study.json",
)

with open(duration_path) as f:
    duration = json.load(f)


epochs = sorted(
    int(epoch)
    for epoch in duration.keys()
)

accuracies = [
    duration[str(epoch)][
        "linear_probe_top1"
    ]
    for epoch in epochs
]


plt.figure(figsize=(7, 5))

plt.plot(
    epochs,
    accuracies,
    marker="o",
)

for epoch, accuracy in zip(
    epochs,
    accuracies,
):
    plt.annotate(
        f"{accuracy:.2f}%",
        (epoch, accuracy),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
    )

plt.xlabel(
    "SimCLR pretraining epochs"
)

plt.ylabel(
    "Linear probe top-1 accuracy (%)"
)

plt.title(
    "Representation Quality vs Pretraining Duration"
)

plt.grid(
    alpha=0.25,
)

save_figure(
    "pretraining_duration.png"
)


# ============================================================
# 4. Augmentation ablation
# ============================================================

augmentation_path = os.path.join(
    RESULTS_DIR,
    "augmentation_ablation.json",
)

with open(augmentation_path) as f:
    augmentation = json.load(f)


augmentation_labels = [
    "Full\naugmentation",
    "No color\njitter",
    "No blur",
    "Weak crop",
]

augmentation_values = [
    augmentation["full_augmentation"],
    augmentation["no_color_jitter"],
    augmentation["no_blur"],
    augmentation["weak_crop"],
]


plt.figure(figsize=(8, 5))

bars = plt.bar(
    augmentation_labels,
    augmentation_values,
)

plt.ylabel(
    "Linear probe top-1 accuracy (%)"
)

plt.title(
    "SimCLR Augmentation Ablation"
)

plt.ylim(0, 60)

plt.bar_label(
    bars,
    fmt="%.2f%%",
    padding=3,
)

save_figure(
    "augmentation_ablation.png"
)


# ============================================================
# 5. Projection-head ablation
# ============================================================

projection_path = os.path.join(
    RESULTS_DIR,
    "projection_head_ablation.json",
)

with open(projection_path) as f:
    projection = json.load(f)


projection_labels = [
    "Without\nprojection head",
    "With\nprojection head",
]

projection_values = [
    projection[
        "without_projection_head"
    ],
    projection[
        "with_projection_head"
    ],
]


plt.figure(figsize=(7, 5))

bars = plt.bar(
    projection_labels,
    projection_values,
)

plt.ylabel(
    "Linear probe top-1 accuracy (%)"
)

plt.title(
    "Effect of the SimCLR Projection Head"
)

plt.ylim(0, 55)

plt.bar_label(
    bars,
    fmt="%.2f%%",
    padding=3,
)

save_figure(
    "projection_head_ablation.png"
)


# ============================================================
# 6. Fine-tuning learning curves
# ============================================================

scratch_history_path = os.path.join(
    RESULTS_DIR,
    "supervised_scratch_history.csv",
)

simclr_history_path = os.path.join(
    RESULTS_DIR,
    "simclr_finetuned_history.csv",
)


scratch_history = pd.read_csv(
    scratch_history_path
)

simclr_history = pd.read_csv(
    simclr_history_path
)


plt.figure(figsize=(8, 5))

plt.plot(
    scratch_history["epoch"],
    scratch_history["val_top1"],
    label="Supervised from scratch",
)

plt.plot(
    simclr_history["epoch"],
    simclr_history["val_top1"],
    label="SimCLR pretrained",
)

plt.xlabel(
    "Fine-tuning epoch"
)

plt.ylabel(
    "Validation top-1 accuracy (%)"
)

plt.title(
    "Fine-Tuning Convergence on Food-101"
)

plt.legend()

plt.grid(
    alpha=0.25,
)

save_figure(
    "finetuning_learning_curves.png"
)


print(
    "\n========================================"
)

print(
    "All project figures generated."
)

print(
    "========================================"
)