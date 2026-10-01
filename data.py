"""MNIST and offline digits data, with training-only printed examples."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import normalize_digit


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    image_size: int = 8


def _tensor_dataset(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
    image_tensor = torch.from_numpy(images).unsqueeze(1)
    label_tensor = torch.from_numpy(labels).long()
    return TensorDataset(image_tensor, label_tensor)


def create_dataloaders(
    batch_size: int = 64, seed: int = 42, dataset: str = "digits",
    data_dir: Path | None = None, font_samples: int = 6000,
) -> DataLoaders:
    """Load MNIST or the default offline stratified 70/15/15 digits splits."""
    if dataset == "mnist":
        return _mnist_dataloaders(batch_size, seed, data_dir, font_samples)
    if dataset != "digits":
        raise ValueError("dataset must be digits or mnist.")
    digits = load_digits()
    images = digits.images.astype(np.float32) / 16.0
    labels = digits.target.astype(np.int64)

    train_images, remaining_images, train_labels, remaining_labels = train_test_split(
        images,
        labels,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    validation_images, test_images, validation_labels, test_labels = train_test_split(
        remaining_images,
        remaining_labels,
        test_size=0.50,
        random_state=seed,
        stratify=remaining_labels,
    )

    generator = torch.Generator().manual_seed(seed)
    return DataLoaders(
        train=DataLoader(
            _tensor_dataset(train_images, train_labels),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        ),
        validation=DataLoader(
            _tensor_dataset(validation_images, validation_labels),
            batch_size=batch_size,
        ),
        test=DataLoader(
            _tensor_dataset(test_images, test_labels),
            batch_size=batch_size,
        ),
    )


def load_mnist(data_dir: Path | None = None) -> tuple[np.ndarray, ...]:
    """Cache the official Keras MNIST archive and verify its SHA-256 checksum."""
    directory = data_dir if data_dir is not None else Path(__file__).parent / "datasets"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "mnist.npz"
    if not path.exists():
        # A partial download never becomes the cached dataset.
        temporary = directory / "mnist.npz.partial"
        with urllib.request.urlopen(MNIST_URL, timeout=60) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != MNIST_SHA256:
            raise ValueError("Downloaded MNIST archive failed its checksum verification.")
        temporary.replace(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != MNIST_SHA256:
        raise ValueError(f"MNIST checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return tuple(archive[key] for key in ("x_train", "y_train", "x_test", "y_test"))


def prepare_training_image(image: np.ndarray, image_size: int = 28) -> np.ndarray:
    """Use the same sizing and centering as external image inference."""
    image = image.copy()
    image[image < 30] = 0
    return normalize_digit(image, image_size)


def make_printed_digits(count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate diverse labeled glyphs; no workspace image is read here."""
    rng = np.random.default_rng(seed)
    fonts = sorted([
        *Path("/usr/share/fonts/dejavu").glob("*.ttf"),
        *Path("/usr/share/fonts/urw-base35").glob("*.otf"),
    ])
    images = np.empty((count, 28, 28), dtype=np.uint8)
    labels = np.arange(count, dtype=np.int64) % 10
    for index, label in enumerate(labels):
        if fonts and rng.random() < 0.6:
            canvas = Image.new("L", (96, 96))
            font = ImageFont.truetype(str(fonts[int(rng.integers(len(fonts)))]), 70)
            ImageDraw.Draw(canvas).text((48, 48), str(label), font=font, fill=255, anchor="mm")
            image = np.asarray(canvas)
        else:
            image = np.zeros((96, 96), dtype=np.uint8)
            face = int(rng.integers(8))
            thickness = int(rng.integers(1, 5))
            cv2.putText(image, str(label), (10, 75), face, 2.2, 255, thickness, cv2.LINE_AA)
        images[index] = prepare_training_image(image)
    return images, labels


def _mnist_dataloaders(
    batch_size: int, seed: int, data_dir: Path | None, font_samples: int,
) -> DataLoaders:
    if font_samples < 0:
        raise ValueError("font_samples must be nonnegative.")
    images, labels, test_images, test_labels = load_mnist(data_dir)
    train_indices, validation_indices = train_test_split(
        np.arange(len(labels)), test_size=0.10, random_state=seed, stratify=labels,
    )
    print("Preparing centered MNIST images and training-only printed digits...", flush=True)
    prepared = np.stack([prepare_training_image(image) for image in images])
    test_prepared = np.stack([prepare_training_image(image) for image in test_images])
    train_images = prepared[train_indices]
    train_labels = labels[train_indices]
    # Retain the original dataset's training subset; its test subset stays held out.
    original = create_dataloaders(seed=seed)
    digit_images, digit_labels = original.train.dataset.tensors
    digit_prepared = np.stack([
        prepare_training_image((image[0].numpy() * 255).round().astype(np.uint8))
        for image in digit_images
    ])
    printed_images, printed_labels = make_printed_digits(font_samples, seed)
    train_images = np.concatenate((train_images, digit_prepared, printed_images))
    train_labels = np.concatenate((train_labels, digit_labels.numpy(), printed_labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoaders(
        train=DataLoader(
            _tensor_dataset(train_images.astype(np.float32) / 255, train_labels),
            batch_size=batch_size, shuffle=True, generator=generator,
        ),
        validation=DataLoader(
            _tensor_dataset(prepared[validation_indices].astype(np.float32) / 255, labels[validation_indices]),
            batch_size=batch_size,
        ),
        test=DataLoader(
            _tensor_dataset(test_prepared.astype(np.float32) / 255, test_labels),
            batch_size=batch_size,
        ),
        image_size=28,
    )
