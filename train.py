"""Train and evaluate the digit CNN."""

import argparse
import copy
import json
import os
import random
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cnn-practice")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import create_dataloaders, prepare_training_image, MNIST_SHA256
from model import DigitCNN


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=("digits", "mnist"), default="mnist")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "datasets")
    parser.add_argument("--font-samples", type=int, default=6000)
    parser.add_argument("--threads", type=int, default=2, help="CPU threads used by PyTorch.")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use CPU, CUDA, or CUDA when available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "artifacts",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    augment: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)
            if augment:
                images = augment_images(images)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


def augment_images(images: torch.Tensor) -> torch.Tensor:
    """Training-only rotation, translation, scale and stroke variation."""
    batch = images.size(0)
    angle = (torch.rand(batch, device=images.device) - 0.5) * 0.4
    scale = 0.85 + torch.rand(batch, device=images.device) * 0.30
    aspect = 0.90 + torch.rand(batch, device=images.device) * 0.20
    transform = torch.zeros(batch, 2, 3, device=images.device)
    transform[:, 0, 0] = angle.cos() * scale * aspect
    transform[:, 0, 1] = -angle.sin() * scale
    transform[:, 1, 0] = angle.sin() * scale * aspect
    transform[:, 1, 1] = angle.cos() * scale
    transform[:, :, 2] = (torch.rand(batch, 2, device=images.device) - 0.5) * 0.18
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, align_corners=False)
    # Mix thick, thin and unchanged strokes in each batch.
    choice = torch.rand(batch, 1, 1, 1, device=images.device)
    thick = F.max_pool2d(images, 3, stride=1, padding=1)
    thin = -F.max_pool2d(-images, 3, stride=1, padding=1)
    images = torch.where(choice < 0.10, thick, torch.where(choice > 0.95, thin, images))
    return images.clamp(0, 1)


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    true_labels: list[np.ndarray] = []
    predicted_labels: list[np.ndarray] = []
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu().numpy()
        true_labels.append(labels.numpy())
        predicted_labels.append(predictions)
    return np.concatenate(true_labels), np.concatenate(predicted_labels)


def save_learning_curves(history: dict[str, list[float]], path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["validation_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["validation_accuracy"], label="Validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        title="Test-set confusion matrix",
        xlabel="Predicted digit",
        ylabel="Actual digit",
        xticks=range(10),
        yticks=range(10),
    )
    for row in range(10):
        for column in range(10):
            axis.text(column, row, matrix[row, column], ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_example(loader: DataLoader, output_dir: Path) -> tuple[Path, int]:
    images, labels = next(iter(loader))
    example = (images[0, 0].numpy() * 255).round().astype(np.uint8)
    label = int(labels[0])
    path = output_dir / f"example_digit_{label}.png"
    cv2.imwrite(str(path), example)
    return path, label


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.learning_rate <= 0:
        raise ValueError("epochs, batch-size, threads and learning-rate must be positive.")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaders = create_dataloaders(
        args.batch_size, args.seed, args.dataset, args.data_dir, args.font_samples,
    )
    model = DigitCNN(image_size=loaders.image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_validation_accuracy = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    print(f"Device: {device}")
    print(f"Train/validation/test: {len(loaders.train.dataset)}/"
          f"{len(loaders.validation.dataset)}/{len(loaders.test.dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders.train, criterion, device, optimizer,
            augment=args.dataset == "mnist" and not args.no_augmentation,
        )
        validation_loss, validation_accuracy = run_epoch(
            model, loaders.validation, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.3f} | "
            f"val loss {validation_loss:.4f}, acc {validation_accuracy:.3f}"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)

    true_labels, predicted_labels = collect_predictions(model, loaders.test, device)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(10)),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(10)))

    checkpoint_path = args.output_dir / "best_model.pt"
    run_config = {
        "dataset": args.dataset, "seed": args.seed, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "epochs": args.epochs,
        "font_samples": args.font_samples if args.dataset == "mnist" else 0,
        "augmentation": args.dataset == "mnist" and not args.no_augmentation,
        "image_size": loaders.image_size,
        "split_sizes": {name: len(getattr(loaders, name).dataset) for name in ("train", "validation", "test")},
        "mnist_sha256": MNIST_SHA256 if args.dataset == "mnist" else None,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": list(range(10)),
            "image_shape": [1, loaders.image_size, loaders.image_size],
            "pixel_range": [0.0, 1.0],
            "preprocessing": "centered-strokes-v2" if args.dataset == "mnist" else "native-digits",
            "run_config": run_config,
        },
        checkpoint_path,
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_config": run_config,
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ("torch", "numpy", "scikit-learn", "opencv-python", "Pillow")}},
        "device": str(device),
        "epochs": args.epochs,
        "best_validation_accuracy": best_validation_accuracy,
        "test_accuracy": report["accuracy"],
        "test_macro_precision": report["macro avg"]["precision"],
        "test_macro_recall": report["macro avg"]["recall"],
        "test_macro_f1": report["macro avg"]["f1-score"],
        "classification_report": report,
        "history": history,
    }
    if args.dataset == "mnist":
        original_test = create_dataloaders(args.batch_size, args.seed).test
        original_images, original_labels = original_test.dataset.tensors
        prepared_images = np.stack([
            prepare_training_image((image[0].numpy() * 255).round().astype(np.uint8))
            for image in original_images
        ])
        original_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(prepared_images.astype(np.float32) / 255).unsqueeze(1),
                original_labels,
            ), batch_size=args.batch_size,
        )
        original_true, original_predicted = collect_predictions(model, original_loader, device)
        summary["original_digits_test_accuracy"] = float((original_true == original_predicted).mean())
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_learning_curves(history, args.output_dir / "learning_curves.png")
    save_confusion_matrix(matrix, args.output_dir / "confusion_matrix.png")
    example_path, example_label = save_example(loaders.test, args.output_dir)

    print("\nBest model evaluation")
    print(f"Accuracy:        {report['accuracy']:.3f}")
    print(f"Macro precision: {report['macro avg']['precision']:.3f}")
    print(f"Macro recall:    {report['macro avg']['recall']:.3f}")
    print(f"Macro F1:        {report['macro avg']['f1-score']:.3f}")
    print(f"Checkpoint:      {checkpoint_path}")
    print(f"Example image:   {example_path} (actual label: {example_label})")


if __name__ == "__main__":
    main()
