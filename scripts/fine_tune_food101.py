import os
import csv
import json
import subprocess

import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Food101
from tqdm import tqdm

from src.models.encoder import ResNet50Encoder
from src.models.simclr import SimCLR


# ============================================================
# Configuration
# ============================================================

batch_size = 128
epochs = 30

learning_rate = 3e-4
weight_decay = 1e-4

num_workers = 4
num_classes = 101

validation_fraction = 0.10

seed = 42


# ============================================================
# Paths
# ============================================================

data_root = "/tmp/food101"

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

checkpoint_root = "/tmp/food101_finetuning"
os.makedirs(checkpoint_root, exist_ok=True)

simclr_checkpoint_path = (
    "/tmp/simclr_epoch100.pt"
)

s3_base = (
    "s3://lachqar/"
    "self-supervised-visual-representation-learning"
)

s3_dataset = (
    f"{s3_base}/datasets/food101/"
)

s3_simclr_checkpoint = (
    f"{s3_base}/checkpoints/"
    "food101_simclr_bs128_100ep/"
    "epoch_100.pt"
)

s3_finetune_root = (
    f"{s3_base}/checkpoints/"
    "food101_finetuning"
)

s3_results_root = (
    f"{s3_base}/results"
)

endpoint = os.environ["AWS_ENDPOINT_URL"]


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
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# Restore Food-101 if necessary
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
        "Restoring Food-101 from S3...\n"
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
        "Food-101 restored.\n"
    )


# ============================================================
# Download SimCLR checkpoint if necessary
# ============================================================

if not os.path.exists(
    simclr_checkpoint_path
):

    print(
        "Downloading SimCLR epoch-100 checkpoint..."
    )

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            s3_simclr_checkpoint,
            simclr_checkpoint_path,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )

    print(
        "SimCLR checkpoint downloaded.\n"
    )


# ============================================================
# Supervised training augmentation
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.2, 1.0),
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Validation / test transform
# ============================================================

eval_transform = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Food-101 datasets
#
# We instantiate the train split twice:
#
# train_dataset -> stochastic training augmentation
# val_dataset   -> deterministic evaluation transform
#
# Both use the same underlying images.
# ============================================================

full_train_dataset = Food101(
    root=data_root,
    split="train",
    transform=train_transform,
    download=False,
)

full_val_dataset = Food101(
    root=data_root,
    split="train",
    transform=eval_transform,
    download=False,
)

test_dataset = Food101(
    root=data_root,
    split="test",
    transform=eval_transform,
    download=False,
)


# ============================================================
# Fixed train / validation split
# ============================================================

dataset_size = len(
    full_train_dataset
)

generator = torch.Generator()

generator.manual_seed(seed)

indices = torch.randperm(
    dataset_size,
    generator=generator,
).tolist()

validation_size = int(
    validation_fraction
    * dataset_size
)

val_indices = indices[
    :validation_size
]

train_indices = indices[
    validation_size:
]


train_dataset = Subset(
    full_train_dataset,
    train_indices,
)

val_dataset = Subset(
    full_val_dataset,
    val_indices,
)


print(
    "Training images:",
    len(train_dataset),
)

print(
    "Validation images:",
    len(val_dataset),
)

print(
    "Test images:",
    len(test_dataset),
)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)


# ============================================================
# Classification model
# ============================================================

class Food101Classifier(nn.Module):

    def __init__(
        self,
        encoder,
        num_classes=101,
    ):

        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Linear(
            encoder.feature_dim,
            num_classes,
        )


    def forward(self, x):

        h = self.encoder(x)

        logits = self.classifier(h)

        return logits


# ============================================================
# Evaluation function
# ============================================================

@torch.inference_mode()
def evaluate(
    model,
    dataloader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    correct_top1 = 0
    correct_top5 = 0

    total = 0


    for images, labels in dataloader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )


        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                device.type == "cuda"
            ),
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )


        batch_size_current = (
            labels.size(0)
        )


        running_loss += (
            loss.item()
            * batch_size_current
        )


        # ----------------------------------------------------
        # Top-1
        # ----------------------------------------------------

        predictions = logits.argmax(
            dim=1
        )

        correct_top1 += (
            predictions == labels
        ).sum().item()


        # ----------------------------------------------------
        # Top-5
        # ----------------------------------------------------

        top5_predictions = (
            logits.topk(
                5,
                dim=1,
            ).indices
        )

        correct_top5 += (
            top5_predictions
            == labels.unsqueeze(1)
        ).any(
            dim=1
        ).sum().item()


        total += batch_size_current


    average_loss = (
        running_loss / total
    )

    top1_accuracy = (
        100.0
        * correct_top1
        / total
    )

    top5_accuracy = (
        100.0
        * correct_top5
        / total
    )


    return (
        average_loss,
        top1_accuracy,
        top5_accuracy,
    )


# ============================================================
# Save history CSV
# ============================================================

def save_history_csv(
    history,
    path,
):

    if len(history) == 0:
        return

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=history[0].keys(),
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# Train one experiment
# ============================================================

def run_experiment(
    experiment_name,
    encoder,
):

    print(
        "\n=============================================="
    )

    print(
        f"EXPERIMENT: {experiment_name}"
    )

    print(
        "==============================================\n"
    )


    # --------------------------------------------------------
    # Entire encoder is trainable
    # --------------------------------------------------------

    for parameter in encoder.parameters():
        parameter.requires_grad = True


    model = Food101Classifier(
        encoder=encoder,
        num_classes=num_classes,
    ).to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


    scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
        ),
    )


    # ========================================================
    # Checkpoint locations
    # ========================================================

    local_experiment_dir = os.path.join(
        checkpoint_root,
        experiment_name,
    )

    os.makedirs(
        local_experiment_dir,
        exist_ok=True,
    )


    latest_path = os.path.join(
        local_experiment_dir,
        "latest.pt",
    )

    best_path = os.path.join(
        local_experiment_dir,
        "best.pt",
    )


    s3_experiment_dir = (
        f"{s3_finetune_root}/"
        f"{experiment_name}"
    )


    s3_latest_path = (
        f"{s3_experiment_dir}/latest.pt"
    )

    s3_best_path = (
        f"{s3_experiment_dir}/best.pt"
    )


    # ========================================================
    # Try restoring existing experiment
    # ========================================================

    start_epoch = 0

    best_val_accuracy = 0.0

    history = []


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


    if result.returncode == 0:

        print(
            "Existing checkpoint found."
        )

        print(
            "Restoring experiment...\n"
        )


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


        best_val_accuracy = checkpoint[
            "best_val_accuracy"
        ]


        history = checkpoint.get(
            "history",
            [],
        )


        print(
            f"Resuming from epoch "
            f"{start_epoch + 1}"
        )

        print(
            f"Best validation accuracy "
            f"so far: "
            f"{best_val_accuracy:.2f}%\n"
        )


    else:

        print(
            "Starting experiment from scratch.\n"
        )


    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        start_epoch,
        epochs,
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0


        progress_bar = tqdm(
            train_loader,
            desc=(
                f"{experiment_name} "
                f"Epoch "
                f"{epoch + 1}/{epochs}"
            ),
        )


        for images, labels in progress_bar:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            # =================================================
            # Forward
            # =================================================

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    device.type == "cuda"
                ),
            ):

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels,
                )


            # =================================================
            # Backward
            # =================================================

            scaler.scale(
                loss
            ).backward()


            scaler.step(
                optimizer
            )


            scaler.update()


            # =================================================
            # Statistics
            # =================================================

            batch_size_current = (
                labels.size(0)
            )


            running_loss += (
                loss.item()
                * batch_size_current
            )


            predictions = logits.argmax(
                dim=1
            )


            correct += (
                predictions
                == labels
            ).sum().item()


            total += (
                batch_size_current
            )


            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=(
                    f"{100.0 * correct / total:.2f}%"
                ),
            )


        # ====================================================
        # Training metrics
        # ====================================================

        train_loss = (
            running_loss
            / total
        )

        train_accuracy = (
            100.0
            * correct
            / total
        )


        # ====================================================
        # Validation
        # ====================================================

        (
            val_loss,
            val_top1,
            val_top5,
        ) = evaluate(
            model,
            val_loader,
            criterion,
        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        print(
            f"\nEpoch {epoch + 1}"
        )

        print(
            f"Train loss : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train acc  : "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Val loss   : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val top-1  : "
            f"{val_top1:.2f}%"
        )

        print(
            f"Val top-5  : "
            f"{val_top5:.2f}%"
        )

        print(
            f"LR         : "
            f"{current_lr:.2e}\n"
        )


        # ====================================================
        # Save metrics
        # ====================================================

        history.append(
            {
                "epoch":
                    epoch + 1,

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss":
                    val_loss,

                "val_top1":
                    val_top1,

                "val_top5":
                    val_top5,

                "learning_rate":
                    current_lr,
            }
        )


        # ====================================================
        # Best model
        # ====================================================

        is_best = (
            val_top1
            > best_val_accuracy
        )


        if is_best:

            best_val_accuracy = (
                val_top1
            )


        # Scheduler AFTER epoch
        scheduler.step()


        # ====================================================
        # Checkpoint
        # ====================================================

        checkpoint = {
            "epoch":
                epoch + 1,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "best_val_accuracy":
                best_val_accuracy,

            "history":
                history,

            "experiment":
                experiment_name,

            "batch_size":
                batch_size,

            "learning_rate":
                learning_rate,

            "weight_decay":
                weight_decay,
        }


        # ====================================================
        # latest.pt
        # ====================================================

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
                s3_latest_path,
                "--endpoint-url",
                endpoint,
            ],
            check=True,
        )


        # ====================================================
        # best.pt
        # ====================================================

        if is_best:

            torch.save(
                checkpoint,
                best_path,
            )


            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    best_path,
                    s3_best_path,
                    "--endpoint-url",
                    endpoint,
                ],
                check=True,
            )


            print(
                "New best validation model saved."
            )


    # ========================================================
    # Download best checkpoint
    # ========================================================

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            s3_best_path,
            best_path,
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )


    best_checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )


    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )


    best_epoch = (
        best_checkpoint["epoch"]
    )


    best_val_accuracy = (
        best_checkpoint[
            "best_val_accuracy"
        ]
    )


    # ========================================================
    # FINAL TEST
    #
    # Test set is touched only here.
    # ========================================================

    print(
        "\nEvaluating best model "
        "on official Food-101 test set...\n"
    )


    (
        test_loss,
        test_top1,
        test_top5,
    ) = evaluate(
        model,
        test_loader,
        criterion,
    )


    print(
        "=============================================="
    )

    print(
        f"{experiment_name} FINAL RESULTS"
    )

    print(
        "=============================================="
    )

    print(
        f"Best epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Best val top-1    : "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Test loss         : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test top-1        : "
        f"{test_top1:.2f}%"
    )

    print(
        f"Test top-5        : "
        f"{test_top5:.2f}%"
    )

    print(
        "==============================================\n"
    )


    # ========================================================
    # Save history locally
    # ========================================================

    history_path = os.path.join(
        results_dir,
        f"{experiment_name}_history.csv",
    )

    save_history_csv(
        best_checkpoint["history"],
        history_path,
    )


    # ========================================================
    # Upload history to S3
    # ========================================================

    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            history_path,
            (
                f"{s3_results_root}/"
                f"{experiment_name}_history.csv"
            ),
            "--endpoint-url",
            endpoint,
        ],
        check=True,
    )


    return {
        "best_epoch":
            best_epoch,

        "best_val_top1":
            best_val_accuracy,

        "test_loss":
            test_loss,

        "test_top1":
            test_top1,

        "test_top5":
            test_top5,
    }


# ============================================================
# EXPERIMENT 1
#
# Supervised ResNet-50 from scratch
# ============================================================

scratch_encoder = (
    ResNet50Encoder()
)


scratch_results = run_experiment(
    experiment_name=(
        "supervised_scratch"
    ),
    encoder=scratch_encoder,
)


del scratch_encoder

torch.cuda.empty_cache()


# ============================================================
# EXPERIMENT 2
#
# SimCLR-pretrained ResNet-50 fine-tuning
# ============================================================

simclr_model = SimCLR(
    projection_dim=128,
    projection_hidden_dim=2048,
)


simclr_checkpoint = torch.load(
    simclr_checkpoint_path,
    map_location="cpu",
    weights_only=False,
)


simclr_model.load_state_dict(
    simclr_checkpoint[
        "model_state_dict"
    ]
)


simclr_encoder = (
    simclr_model.encoder
)


# Projection head is discarded.
del simclr_model


simclr_results = run_experiment(
    experiment_name=(
        "simclr_finetuned"
    ),
    encoder=simclr_encoder,
)


# ============================================================
# Final comparison
# ============================================================

improvement = (
    simclr_results["test_top1"]
    - scratch_results["test_top1"]
)


final_results = {
    "supervised_from_scratch":
        scratch_results,

    "simclr_pretrained_finetuned":
        simclr_results,

    "simclr_improvement_percentage_points":
        improvement,

    "epochs":
        epochs,

    "batch_size":
        batch_size,

    "learning_rate":
        learning_rate,

    "weight_decay":
        weight_decay,

    "validation_fraction":
        validation_fraction,

    "seed":
        seed,
}


# ============================================================
# Save JSON locally
# ============================================================

results_json_path = os.path.join(
    results_dir,
    "food101_finetuning_comparison.json",
)


with open(
    results_json_path,
    "w",
) as f:

    json.dump(
        final_results,
        f,
        indent=4,
    )


# ============================================================
# Upload JSON to S3
# ============================================================

subprocess.run(
    [
        "aws",
        "s3",
        "cp",
        results_json_path,
        (
            f"{s3_results_root}/"
            "food101_finetuning_comparison.json"
        ),
        "--endpoint-url",
        endpoint,
    ],
    check=True,
)


# ============================================================
# Print comparison
# ============================================================

print(
    "\n===================================================="
)

print(
    "FOOD-101 FINE-TUNING COMPARISON"
)

print(
    "===================================================="
)


print(
    "Supervised from scratch"
)

print(
    f"Test top-1 : "
    f"{scratch_results['test_top1']:.2f}%"
)

print(
    f"Test top-5 : "
    f"{scratch_results['test_top5']:.2f}%"
)


print(
    "\nSimCLR pretrained + fine-tuned"
)

print(
    f"Test top-1 : "
    f"{simclr_results['test_top1']:.2f}%"
)

print(
    f"Test top-5 : "
    f"{simclr_results['test_top5']:.2f}%"
)


print(
    "\nSimCLR improvement:"
)

print(
    f"{improvement:+.2f} percentage points"
)


print(
    "===================================================="
)

print(
    f"\nResults saved to:"
)

print(
    results_json_path
)