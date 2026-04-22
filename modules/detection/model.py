import torch
import torch.nn as nn


class MalTwinCNN(nn.Module):
    """
    Custom CNN for grayscale malware image classification.

    Input shape: (batch, 1, 128, 128)
    Output shape: (batch, num_classes) — raw logits

    SRS ref: FE-1 of Module 5
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 1 → 32 channels, output (batch, 32, 64, 64)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Block 2: 32 → 64 channels, output (batch, 64, 32, 32)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Block 3: 64 → 128 channels, output (batch, 128, 16, 16)
        self.block3_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.block3_bn1 = nn.BatchNorm2d(128)
        self.block3_relu1 = nn.ReLU(inplace=True)
        self.block3_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.block3_bn2 = nn.BatchNorm2d(128)
        self.block3_relu2 = nn.ReLU(inplace=True)
        self.block3_pool = nn.MaxPool2d(2, 2)
        self.block3_drop = nn.Dropout2d(0.25)

        # Reference to the final conv layer for Grad-CAM (Module 7)
        self.gradcam_layer = self.block3_conv2

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_pool(x)
        x = self.block3_drop(x)
        x = self.classifier(x)
        return x
