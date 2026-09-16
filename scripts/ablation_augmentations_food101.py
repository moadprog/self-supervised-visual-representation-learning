import os
import json
import subprocess

import torch

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import Food101
from tqdm import tqdm

from src.data.augmentations import TwoCropsTransform
from src.models.simclr import SimCLR
from src.losses.nt_xent import NTXentLoss

from src.evaluate.linear_probe_utils import (
    get_food101_eval_loaders,
    linear_probe_accuracy,
)


# ============================================================
# Configuration
# ============================================================

batch_size = 128

ablation_epochs = 30

# IMPORTANT:
# Keep T_max=100 so LR evolution during first 30 epochs
# matches the original baseline run.
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

s3_baseline_checkpoint = (
    f"{s3_base}/checkpoints/"
    "food101_simclr_bs128_100ep/"
    "epoch_030.pt"
)

s3_ablation_root = (
    f"{s3_base}/checkpoints/"
    "augmentation_ablation"
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
# Augmentation factory
# ============================================================

def get_ablation_transform(
    variant,
):

    color_jitter = (
        transforms.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
        )
    )

    if variant == "weak_crop":

        crop_scale = (
            0.6,
            1.0,
        )

    else:

        crop_scale = (
            0.2,
            1.0,
        )


    operations = [
        transforms.RandomResizedCrop(
            224,
            scale=crop_scale,
        ),

        transforms.RandomHorizontalFlip(),
    ]


    # --------------------------------------------------------
    # Color jitter
    # --------------------------------------------------------

    if variant != "no_color_jitter":

        operations.append(
            transforms.RandomApply(
                [color_jitter],
                p=0.8,
            )
        )


    # --------------------------------------------------------
    # Grayscale stays unchanged
    # --------------------------------------------------------

    operations.append(
        transforms.RandomGrayscale(
            p=0.2
        )
    )


    # --------------------------------------------------------
    # Gaussian blur
    #
    # Same behavior as our original pipeline:
    # blur is always applied.
    # --------------------------------------------------------

    if variant != "no_blur":

        operations.append(
            transforms.GaussianBlur(
                kernel_size=9,
                sigma=(0.1, 2.0),
            )
        )


    operations.extend([
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],

            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


    transform = transforms.Compose(
        operations
    )


    return TwoCropsTransform(
        transform
    )


# ============================================================
# Train one augmentation variant
# ============================================================

def train_variant(
    variant,
):

    print(
        "\n========================================"
    )

    print(
        f"AUGMENTATION ABLATION: {variant}"
    )

    print(
        "========================================\n"
    )


    torch.manual_seed(seed)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = Food101(
        root=data_root,
        split="train",
        transform=(
            get_ablation_transform(
                variant
            )
        ),
        download=False,
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


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = SimCLR(
        projection_dim=128,
        projection_hidden_dim=2048,
    ).to(device)


    criterion = NTXentLoss(
        temperature=temperature
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
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


    # --------------------------------------------------------
    # Resume paths
    # --------------------------------------------------------

    local_dir = (
        f"/tmp/"
        f"augmentation_ablation/"
        f"{variant}"
    )

    os.makedirs(
        local_dir,
        exist_ok=True,
    )


    latest_path = (
        f"{local_dir}/latest.pt"
    )


    s3_latest = (
        f"{s3_ablation_root}/"
        f"{variant}/latest.pt"
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

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
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

        start_epoch = checkpoint[
            "epoch"
        ]

        print(
            f"Resuming from epoch "
            f"{start_epoch + 1}"
        )


    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        ablation_epochs,
    ):

        model.train()

        running_loss = 0.0


        progress = tqdm(
            loader,
            desc=(
                f"{variant} "
                f"{epoch + 1}/"
                f"{ablation_epochs}"
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

                _, z1 = model(x1)

                _, z2 = model(x2)

                loss = criterion(
                    z1,
                    z2,
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

            "variant":
                variant,

            "model_state_dict":
                model.state_dict(),

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


    # --------------------------------------------------------
    # Save final epoch checkpoint
    # --------------------------------------------------------

    final_path = (
        f"{local_dir}/"
        f"epoch_{ablation_epochs:03d}.pt"
    )


    torch.save(
        checkpoint,
        final_path,
    )


    s3_final = (
        f"{s3_ablation_root}/"
        f"{variant}/"
        f"epoch_{ablation_epochs:03d}.pt"
    )


    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            final_path,
            s3_final,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )


    encoder = model.encoder

    del model


    return encoder


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


results = {}


# ============================================================
# Baseline: original epoch 30
# ============================================================

print(
    "\nEvaluating original "
    "augmentation baseline...\n"
)


baseline_path = (
    "/tmp/"
    "simclr_baseline_epoch030.pt"
)


subprocess.run(
    [
        "aws",
        "s3",
        "cp",
        s3_baseline_checkpoint,
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


baseline_probe = linear_probe_accuracy(
    encoder=baseline_model.encoder,
    train_loader=eval_train_loader,
    test_loader=eval_test_loader,
    device=device,
    name="full_augmentation_epoch30",
)


results["full_augmentation"] = (
    baseline_probe[
        "best_test_top1"
    ]
)


del baseline_model


# ============================================================
# Ablations
# ============================================================

variants = [
    "no_color_jitter",
    "no_blur",
    "weak_crop",
]


for variant in variants:

    encoder = train_variant(
        variant
    )


    probe = linear_probe_accuracy(
        encoder=encoder,
        train_loader=eval_train_loader,
        test_loader=eval_test_loader,
        device=device,
        name=variant,
    )


    results[variant] = (
        probe[
            "best_test_top1"
        ]
    )


    del encoder

    if device.type == "cuda":
        torch.cuda.empty_cache()


# ============================================================
# Save results
# ============================================================

results_path = (
    "results/"
    "augmentation_ablation.json"
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
            "augmentation_ablation.json"
        ),
        "--endpoint-url",
        endpoint,
    ],
    check=True,
)


# ============================================================
# Final comparison
# ============================================================

print(
    "\n========================================"
)

print(
    "AUGMENTATION ABLATION RESULTS"
)

print(
    "========================================"
)


for name, accuracy in results.items():

    print(
        f"{name:20s}: "
        f"{accuracy:.2f}%"
    )