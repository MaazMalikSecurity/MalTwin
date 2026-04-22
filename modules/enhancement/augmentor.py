import torch
import torchvision.transforms as T


class GaussianNoise:
    """
    Custom torchvision-compatible transform.
    Operates on torch.Tensor post ToTensor().
    Adds Gaussian noise with mean=0, std sampled uniformly from [0.01, 0.05].
    Clamps output to [0.0, 1.0].
    SRS ref: FE-2 of Module 4
    """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        std = torch.empty(1).uniform_(0.01, 0.05).item()
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0.0, 1.0)


def get_train_transforms(img_size: int) -> T.Compose:
    """
    Returns a Compose pipeline for TRAINING data only.
    NOTE: Our dataset returns tensors already (shape 1,H,W), so we skip ToTensor()
    and apply transforms directly on tensors.
    SRS ref: FE-1 of Module 4
    """
    return T.Compose([
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.ColorJitter(brightness=0.2),
        GaussianNoise(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transforms(img_size: int) -> T.Compose:
    """
    Returns Compose for val/test — NO augmentation, only normalize.
    SRS ref: FE-1 of Module 4
    """
    return T.Compose([
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
