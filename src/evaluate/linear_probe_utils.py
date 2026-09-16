import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import Food101
from tqdm import tqdm


# ============================================================
# Evaluation transform
# ============================================================

def get_food101_eval_loaders(
    data_root="/tmp/food101",
    feature_batch_size=128,
    num_workers=4,
):

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = Food101(
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=feature_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=feature_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, test_loader


# ============================================================
# Frozen feature extraction
# ============================================================

@torch.inference_mode()
def extract_features(
    encoder,
    dataloader,
    device,
    description,
):

    encoder.eval()

    features_list = []
    labels_list = []

    for images, labels in tqdm(
        dataloader,
        desc=description,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        features = encoder(images)

        features_list.append(
            features.cpu()
        )

        labels_list.append(
            labels.cpu()
        )

    features = torch.cat(
        features_list,
        dim=0,
    )

    labels = torch.cat(
        labels_list,
        dim=0,
    )

    return features, labels


# ============================================================
# Linear probe
# ============================================================

def linear_probe_accuracy(
    encoder,
    train_loader,
    test_loader,
    device,
    linear_epochs=30,
    linear_batch_size=512,
    learning_rate=0.1,
    num_classes=101,
    seed=42,
    name="encoder",
):

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    encoder = encoder.to(device)

    # --------------------------------------------------------
    # Extract representations once
    # --------------------------------------------------------

    train_features, train_labels = extract_features(
        encoder,
        train_loader,
        device,
        f"{name} - train features",
    )

    test_features, test_labels = extract_features(
        encoder,
        test_loader,
        device,
        f"{name} - test features",
    )

    # Encoder is no longer needed on GPU
    encoder = encoder.cpu()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Feature datasets
    # --------------------------------------------------------

    train_feature_dataset = TensorDataset(
        train_features,
        train_labels,
    )

    test_feature_dataset = TensorDataset(
        test_features,
        test_labels,
    )

    train_feature_loader = DataLoader(
        train_feature_dataset,
        batch_size=linear_batch_size,
        shuffle=True,
    )

    test_feature_loader = DataLoader(
        test_feature_dataset,
        batch_size=linear_batch_size,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Linear classifier
    # --------------------------------------------------------

    torch.manual_seed(seed)

    classifier = nn.Linear(
        train_features.shape[1],
        num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=linear_epochs,
        eta_min=1e-4,
    )

    best_test_accuracy = 0.0

    history = []

    # --------------------------------------------------------
    # Probe training
    # --------------------------------------------------------

    for epoch in range(linear_epochs):

        classifier.train()

        correct = 0
        total = 0
        running_loss = 0.0

        for features, labels in train_feature_loader:

            features = features.to(
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

            logits = classifier(features)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        scheduler.step()

        train_accuracy = (
            100.0 * correct / total
        )

        train_loss = (
            running_loss / total
        )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        classifier.eval()

        test_correct = 0
        test_total = 0

        with torch.inference_mode():

            for features, labels in test_feature_loader:

                features = features.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                logits = classifier(features)

                predictions = logits.argmax(
                    dim=1
                )

                test_correct += (
                    predictions == labels
                ).sum().item()

                test_total += labels.size(0)

        test_accuracy = (
            100.0
            * test_correct
            / test_total
        )

        best_test_accuracy = max(
            best_test_accuracy,
            test_accuracy,
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        })

        print(
            f"{name} "
            f"| probe epoch "
            f"{epoch + 1:02d}/{linear_epochs} "
            f"| train={train_accuracy:.2f}% "
            f"| test={test_accuracy:.2f}%"
        )

    return {
        "best_test_top1": best_test_accuracy,
        "history": history,
    }