from torchvision import transforms


class TwoCropsTransform:
    """
    Apply the same augmentation pipeline twice independently
    to produce the two positive views required by SimCLR.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        view1 = self.base_transform(x)
        view2 = self.base_transform(x)

        return view1, view2


def get_simclr_transform(image_size=96):
    """
    SimCLR-style augmentations for STL-10.
    """

    color_jitter = transforms.ColorJitter(
        brightness=0.8,
        contrast=0.8,
        saturation=0.8,
        hue=0.2,
    )

    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.2, 1.0),
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomApply(
            [color_jitter],
            p=0.8,
        ),

        transforms.RandomGrayscale(
            p=0.2,
        ),

        transforms.GaussianBlur(
            kernel_size=9,
            sigma=(0.1, 2.0),
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return TwoCropsTransform(transform)