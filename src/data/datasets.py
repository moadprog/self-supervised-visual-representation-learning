from torchvision.datasets import STL10, Food101

from src.data.augmentations import get_simclr_transform


def get_stl10_pretrain_dataset(
    root="data",
    download=True,
    image_size=96,
):
    """
    STL-10 unlabeled split for self-supervised SimCLR pretraining.
    """

    transform = get_simclr_transform(image_size=image_size)

    dataset = STL10(
        root=root,
        split="unlabeled",
        transform=transform,
        download=download,
    )

    return dataset

def get_food101_pretrain_dataset(
    root="/tmp/food101",
    download=True,
    image_size=224,
):
    """
    Food-101 training split for self-supervised SimCLR pretraining.

    Labels are returned by the dataset but ignored during SSL training.
    """

    transform = get_simclr_transform(
        image_size=image_size
    )

    dataset = Food101(
        root=root,
        split="train",
        transform=transform,
        download=download,
    )

    return dataset