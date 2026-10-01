"""CNN architecture for the handwritten-digit practice project."""

from torch import Tensor, nn


class DigitCNN(nn.Module):
    """A small convolutional network for square grayscale images."""

    def __init__(self, num_classes: int = 10, image_size: int = 8) -> None:
        super().__init__()
        if image_size < 8 or image_size % 4:
            raise ValueError("image_size must be at least 8 and divisible by 4.")
        self.image_size = image_size
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (image_size // 4) ** 2, 64),
            nn.ReLU(),
            nn.Dropout(p=0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: Tensor) -> Tensor:
        features = self.features(images)
        return self.classifier(features)
