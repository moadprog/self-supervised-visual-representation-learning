import os
import json
import subprocess

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import (
    get_food101_pretrain_dataset
)

from src.models.encoder import (
    ResNet50Encoder
)

from src.models.simclr import SimCLR

from src.losses.nt_xent import (
    NTXentLoss
)

from src.evaluate.linear_probe_utils import (
    get_food101_eval_loaders,
    linear_probe_accuracy,
)


# ============================================================
# Configuration
# ============================================================

batch_size = 128

epochs = 30

# Match first 30 epochs of original 100-epoch run
scheduler_t_max = 100

learning_rate = 3e-4
weight_decay = 1e-4

temperature = 0.5

num_workers = 4

seed = 42


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


endpoint = os.environ[
    "AWS_ENDPOINT_URL"
]


# ============================================================
# Paths
# ============================================================

data_root = "/tmp/food101"

s3_base = (
    "s3://lachqar/"
    "self-supervised-visual-representation-learning"
)

s3_dataset = (
    f"{s3_base}/datasets/food101/"
)

s3_baseline = (
    f"{s3_base}/checkpoints/"
    "food101_simclr_bs128_100ep/"
    "epoch_030.pt"
)

s3_no_head_dir = (
    f"{s3_base}/checkpoints/"
    "projection_head_ablation/"
    "no_projection_head"
)

results_dir = "results"

os.makedirs(
    results_dir,
    exist_ok=True,
)


# ============================================================
# Restore dataset
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
# Dataset
# ============================================================

dataset = (
    get_food101_pretrain_dataset(
        root=data_root,
        download=False,
        image_size=224,
    )
)


loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)


# ============================================================
# Model WITHOUT projection head
# ============================================================

torch.manual_seed(seed)

if device.type == "cuda":
    torch.cuda.manual_seed_all(seed)


encoder = ResNet50Encoder().to(
    device
)


criterion = NTXentLoss(
    temperature=temperature
)


optimizer = torch.optim.AdamW(
    encoder.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)


scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=1e-6,
    )
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(
        device.type == "cuda"
    ),
)


# ============================================================
# Resume paths
# ============================================================

local_dir = (
    "/tmp/"
    "projection_head_ablation"
)

os.makedirs(
    local_dir,
    exist_ok=True,
)


latest_path = (
    f"{local_dir}/latest.pt"
)


s3_latest = (
    f"{s3_no_head_dir}/latest.pt"
)


start_epoch = 0


result = subprocess.run(
    [
        "aws",
        "s3",
        "cp",
        s3_latest,
        latest_path,
        "--endpoint-url",
        endpoint,
    ],
    capture_output=True,
    text=True,
)


if result.returncode == 0:

    checkpoint = torch.load(
        latest_path,
        map_location=device,
        weights_only=False,
    )

    encoder.load_state_dict(
        checkpoint[
            "encoder_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    scaler.load_state_dict(
        checkpoint[
            "scaler_state_dict"
        ]
    )

    start_epoch = (
        checkpoint["epoch"]
    )

    print(
        f"Resuming from epoch "
        f"{start_epoch + 1}"
    )


# ============================================================
# Training WITHOUT projector
# ============================================================

for epoch in range(
    start_epoch,
    epochs,
):

    encoder.train()

    running_loss = 0.0


    progress = tqdm(
        loader,
        desc=(
            f"No projector "
            f"{epoch + 1}/{epochs}"
        ),
    )


    for views, _ in progress:

        x1, x2 = views


        x1 = x1.to(
            device,
            non_blocking=True,
        )

        x2 = x2.to(
            device,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                device.type
                == "cuda"
            ),
        ):

            # -----------------------------------------------
            # IMPORTANT:
            # Contrastive loss directly on h.
            # No projection MLP.
            # -----------------------------------------------

            h1 = encoder(x1)

            h2 = encoder(x2)


            loss = criterion(
                h1,
                h2,
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        running_loss += (
            loss.item()
        )


        progress.set_postfix(
            loss=(
                f"{loss.item():.4f}"
            )
        )


    average_loss = (
        running_loss
        / len(loader)
    )


    scheduler.step()


    checkpoint = {
        "epoch":
            epoch + 1,

        "encoder_state_dict":
            encoder.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "loss":
            average_loss,
    }


    torch.save(
        checkpoint,
        latest_path,
    )


    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            latest_path,
            s3_latest,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )


    print(
        f"Epoch {epoch + 1}: "
        f"loss={average_loss:.4f}"
    )


# ============================================================
# Save final checkpoint
# ============================================================

final_path = (
    f"{local_dir}/"
    "epoch_030.pt"
)


torch.save(
    checkpoint,
    final_path,
)


subprocess.run(
    [
        "aws",
        "s3",
        "cp",
        final_path,
        (
            f"{s3_no_head_dir}/"
            "epoch_030.pt"
        ),
        "--endpoint-url",
        endpoint,
    ],
    check=True,
)


# ============================================================
# Evaluation loaders
# ============================================================

eval_train_loader, eval_test_loader = (
    get_food101_eval_loaders(
        data_root=data_root,
        feature_batch_size=128,
        num_workers=4,
    )
)


# ============================================================
# Linear probe: NO projector
# ============================================================

no_head_probe = (
    linear_probe_accuracy(
        encoder=encoder,
        train_loader=eval_train_loader,
        test_loader=eval_test_loader,
        device=device,
        name="no_projection_head",
    )
)


no_head_accuracy = (
    no_head_probe[
        "best_test_top1"
    ]
)


# ============================================================
# Linear probe: ORIGINAL projector baseline at epoch 30
# ============================================================

baseline_path = (
    "/tmp/"
    "baseline_projection_epoch030.pt"
)


subprocess.run(
    [
        "aws",
        "s3",
        "cp",
        s3_baseline,
        baseline_path,
        "--endpoint-url",
        endpoint,
    ],
    check=True,
)


baseline_model = SimCLR(
    projection_dim=128,
    projection_hidden_dim=2048,
)


baseline_checkpoint = torch.load(
    baseline_path,
    map_location="cpu",
    weights_only=False,
)


baseline_model.load_state_dict(
    baseline_checkpoint[
        "model_state_dict"
    ]
)


baseline_probe = (
    linear_probe_accuracy(
        encoder=baseline_model.encoder,
        train_loader=eval_train_loader,
        test_loader=eval_test_loader,
        device=device,
        name="with_projection_head",
    )
)


baseline_accuracy = (
    baseline_probe[
        "best_test_top1"
    ]
)


# ============================================================
# Results
# ============================================================

results = {
    "with_projection_head":
        baseline_accuracy,

    "without_projection_head":
        no_head_accuracy,

    "projection_head_gain":
        (
            baseline_accuracy
            - no_head_accuracy
        ),
}


results_path = (
    "results/"
    "projection_head_ablation.json"
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


subprocess.run(
    [
        "aws",
        "s3",
        "cp",
        results_path,
        (
            f"{s3_base}/results/"
            "projection_head_ablation.json"
        ),
        "--endpoint-url",
        endpoint,
    ],
    check=True,
)


print(
    "\n========================================"
)

print(
    "PROJECTION HEAD ABLATION"
)

print(
    "========================================"
)


print(
    f"With projection head    : "
    f"{baseline_accuracy:.2f}%"
)


print(
    f"Without projection head : "
    f"{no_head_accuracy:.2f}%"
)


print(
    f"Projection-head gain    : "
    f"{baseline_accuracy - no_head_accuracy:+.2f} pp"
)