import os
import json
import subprocess

import torch

from src.models.simclr import SimCLR
from src.evaluate.linear_probe_utils import (
    get_food101_eval_loaders,
    linear_probe_accuracy,
)


# ============================================================
# Configuration
# ============================================================

epochs_to_evaluate = [
    10,
    20,
    50,
    100,
]

data_root = "/tmp/food101"

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

endpoint = os.environ[
    "AWS_ENDPOINT_URL"
]

s3_base = (
    "s3://lachqar/"
    "self-supervised-visual-representation-learning"
)

s3_dataset = (
    f"{s3_base}/datasets/food101/"
)

s3_checkpoint_dir = (
    f"{s3_base}/checkpoints/"
    "food101_simclr_bs128_100ep"
)

results_dir = "results"

os.makedirs(
    results_dir,
    exist_ok=True,
)


# ============================================================
# Restore Food-101 if necessary
# ============================================================

if not os.path.exists(
    "/tmp/food101/food-101"
):

    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            s3_dataset,
            data_root,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )


# ============================================================
# Evaluation loaders
# ============================================================

train_loader, test_loader = (
    get_food101_eval_loaders(
        data_root=data_root,
        feature_batch_size=128,
        num_workers=4,
    )
)


# ============================================================
# Evaluate each checkpoint
# ============================================================

results = {}


for pretrain_epoch in epochs_to_evaluate:

    print(
        "\n========================================"
    )

    print(
        f"PRETRAINING EPOCH {pretrain_epoch}"
    )

    print(
        "========================================\n"
    )

    local_checkpoint = (
        f"/tmp/simclr_"
        f"epoch_{pretrain_epoch:03d}.pt"
    )

    s3_checkpoint = (
        f"{s3_checkpoint_dir}/"
        f"epoch_{pretrain_epoch:03d}.pt"
    )

    # --------------------------------------------------------
    # Download checkpoint
    # --------------------------------------------------------

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            s3_checkpoint,
            local_checkpoint,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )

    # --------------------------------------------------------
    # Load SimCLR
    # --------------------------------------------------------

    model = SimCLR(
        projection_dim=128,
        projection_hidden_dim=2048,
    )

    checkpoint = torch.load(
        local_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    encoder = model.encoder

    del model

    # --------------------------------------------------------
    # Linear probe
    # --------------------------------------------------------

    probe_result = linear_probe_accuracy(
        encoder=encoder,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        linear_epochs=30,
        linear_batch_size=512,
        learning_rate=0.1,
        name=(
            f"SimCLR epoch "
            f"{pretrain_epoch}"
        ),
    )

    results[str(pretrain_epoch)] = {
        "linear_probe_top1":
            probe_result[
                "best_test_top1"
            ]
    }

    print(
        f"\nEpoch {pretrain_epoch} "
        f"linear probe: "
        f"{probe_result['best_test_top1']:.2f}%"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# Save results
# ============================================================

results_path = (
    "results/"
    "training_duration_study.json"
)

with open(
    results_path,
    "w",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )


# ============================================================
# Upload to S3
# ============================================================

subprocess.run(
    [
        "aws",
        "s3",
        "cp",
        results_path,
        (
            f"{s3_base}/results/"
            "training_duration_study.json"
        ),
        "--endpoint-url",
        endpoint,
    ],
    check=True,
)


# ============================================================
# Final summary
# ============================================================

print(
    "\n========================================"
)

print(
    "TRAINING DURATION STUDY"
)

print(
    "========================================"
)

for epoch, values in results.items():

    print(
        f"Epoch {epoch:>3}: "
        f"{values['linear_probe_top1']:.2f}%"
    )