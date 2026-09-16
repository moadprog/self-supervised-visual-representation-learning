import os
import subprocess

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.datasets import get_food101_pretrain_dataset
from src.models.simclr import SimCLR
from src.losses.nt_xent import NTXentLoss


# ============================================================
# Configuration
# ============================================================

batch_size = 128
epochs = 100

learning_rate = 3e-4
weight_decay = 1e-4
temperature = 0.5

num_workers = 4
seed = 42

experiment_name = "food101_simclr_bs128_100ep"


# ============================================================
# Paths
# ============================================================

data_root = "/tmp/food101"

checkpoint_dir = (
    f"/tmp/simclr_checkpoints/{experiment_name}"
)

tensorboard_dir = (
    f"/tmp/tensorboard/{experiment_name}"
)

s3_base = (
    "s3://lachqar/"
    "self-supervised-visual-representation-learning"
)

s3_dataset = (
    f"{s3_base}/datasets/food101/"
)

s3_checkpoint_dir = (
    f"{s3_base}/checkpoints/{experiment_name}"
)

s3_tensorboard_dir = (
    f"{s3_base}/tensorboard/{experiment_name}"
)

endpoint = os.environ["AWS_ENDPOINT_URL"]

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Restore Food-101 from S3 if /tmp is empty
# ============================================================

food101_folder = os.path.join(
    data_root,
    "food-101",
)

if not os.path.exists(food101_folder):

    print(
        "\nFood-101 not found locally."
    )

    print(
        "Restoring dataset from S3...\n"
    )

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

    print(
        "Food-101 restored successfully.\n"
    )


# ============================================================
# Dataset
# ============================================================

dataset = get_food101_pretrain_dataset(
    root=data_root,
    download=False,
    image_size=224,
)

print(
    "Number of training images:",
    len(dataset),
)


# ============================================================
# DataLoader
# ============================================================

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)


print(
    "Batches per epoch:",
    len(dataloader),
)


# ============================================================
# Model
# ============================================================

model = SimCLR(
    projection_dim=128,
    projection_hidden_dim=2048,
).to(device)


# ============================================================
# NT-Xent loss
# ============================================================

criterion = NTXentLoss(
    temperature=temperature
)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
)


# ============================================================
# Cosine learning-rate scheduler
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=1e-6,
)


# ============================================================
# Automatic mixed precision
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(device.type == "cuda"),
)


# ============================================================
# TensorBoard
# ============================================================

writer = SummaryWriter(
    log_dir=tensorboard_dir
)


# ============================================================
# Checkpoint paths
# ============================================================

latest_path = os.path.join(
    checkpoint_dir,
    "latest.pt",
)

s3_latest_path = (
    f"{s3_checkpoint_dir}/latest.pt"
)


# ============================================================
# Try restoring latest checkpoint from S3
# ============================================================

print(
    "\nChecking S3 for an existing checkpoint..."
)

result = subprocess.run(
    [
        "aws",
        "s3",
        "cp",
        s3_latest_path,
        latest_path,
        "--endpoint-url",
        endpoint,
    ],
    capture_output=True,
    text=True,
)


# ============================================================
# Resume training if checkpoint exists
# ============================================================

start_epoch = 0

if result.returncode == 0:

    print(
        "Existing checkpoint found."
    )

    print(
        "Restoring training state...\n"
    )

    checkpoint = torch.load(
        latest_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    print(
        f"Training restored after epoch "
        f"{start_epoch}."
    )

    print(
        f"Next epoch: {start_epoch + 1}\n"
    )

else:

    print(
        "No existing real-training checkpoint found."
    )

    print(
        "Starting training from scratch.\n"
    )


# ============================================================
# Global TensorBoard step
# ============================================================

global_step = (
    start_epoch * len(dataloader)
)


# ============================================================
# Training loop
# ============================================================

for epoch in range(
    start_epoch,
    epochs,
):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{epochs}",
    )


    # --------------------------------------------------------
    # Mini-batch loop
    # --------------------------------------------------------

    for views, _ in progress_bar:

        x1, x2 = views

        x1 = x1.to(
            device,
            non_blocking=True,
        )

        x2 = x2.to(
            device,
            non_blocking=True,
        )


        # Clear previous gradients
        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward pass with mixed precision
        # ----------------------------------------------------

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                device.type == "cuda"
            ),
        ):

            _, z1 = model(x1)
            _, z2 = model(x2)

            loss = criterion(
                z1,
                z2,
            )


        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        scaler.scale(
            loss
        ).backward()


        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

        scaler.step(
            optimizer
        )

        scaler.update()


        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += loss.item()

        writer.add_scalar(
            "Loss/train_step",
            loss.item(),
            global_step,
        )

        global_step += 1

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )


    # ========================================================
    # End of epoch
    # ========================================================

    average_loss = (
        running_loss
        / len(dataloader)
    )

    print(
        f"\nEpoch {epoch + 1}: "
        f"average loss = "
        f"{average_loss:.4f}"
    )


    # --------------------------------------------------------
    # TensorBoard epoch statistics
    # --------------------------------------------------------

    writer.add_scalar(
        "Loss/train_epoch",
        average_loss,
        epoch + 1,
    )

    writer.add_scalar(
        "LearningRate",
        optimizer.param_groups[0]["lr"],
        epoch + 1,
    )


    # --------------------------------------------------------
    # Update cosine LR
    # --------------------------------------------------------

    scheduler.step()


    # --------------------------------------------------------
    # Build checkpoint
    # --------------------------------------------------------

    checkpoint = {
        "epoch": epoch + 1,

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

        "batch_size":
            batch_size,

        "learning_rate":
            learning_rate,

        "temperature":
            temperature,

        "weight_decay":
            weight_decay,
    }


    # ========================================================
    # Save latest checkpoint locally
    # ========================================================

    torch.save(
        checkpoint,
        latest_path,
    )


    # ========================================================
    # Upload latest checkpoint to S3
    # ========================================================

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            latest_path,
            s3_latest_path,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )

    print(
        f"Latest checkpoint uploaded: "
        f"epoch {epoch + 1}"
    )


    # ========================================================
    # Permanent snapshot every 10 epochs
    # ========================================================

    if (
        (epoch + 1) % 10 == 0
        or
        (epoch + 1) == epochs
    ):

        snapshot_name = (
            f"epoch_{epoch + 1:03d}.pt"
        )

        snapshot_path = os.path.join(
            checkpoint_dir,
            snapshot_name,
        )

        torch.save(
            checkpoint,
            snapshot_path,
        )

        s3_snapshot_path = (
            f"{s3_checkpoint_dir}/"
            f"{snapshot_name}"
        )

        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                snapshot_path,
                s3_snapshot_path,
                "--endpoint-url",
                endpoint,
            ],
            check=True,
        )

        print(
            f"Permanent snapshot saved: "
            f"{snapshot_name}"
        )


    # ========================================================
    # Flush TensorBoard
    # ========================================================

    writer.flush()


    # ========================================================
    # Backup TensorBoard logs to S3
    # ========================================================

    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            tensorboard_dir,
            s3_tensorboard_dir,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )


    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"complete.\n"
    )


# ============================================================
# End training
# ============================================================

writer.close()

print(
    "\n======================================"
)

print(
    "SimCLR pretraining completed."
)

print(
    "======================================"
)

