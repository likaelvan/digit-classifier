"""Run the trained CNN on a digit image, using OpenCV for preprocessing."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from model import DigitCNN
from preprocessing import extract_digit, extract_digits, normalize_digit


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_DIR / "artifacts" / "best_model.pt",
    )
    parser.add_argument(
        "--invert",
        choices=("auto", "yes", "no"),
        default="auto",
        help="The model expects a bright digit on a dark background.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--number", action="store_true", help="Read one row of separated digits as an unsigned integer string.")
    parser.add_argument("--save-preprocessed", type=Path)
    return parser.parse_args()


def _border_mean(image: np.ndarray) -> float:
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    return float(border.mean())


def preprocess_image(
    path: Path, invert: str = "auto", image_size: int = 8,
) -> tuple[torch.Tensor, np.ndarray]:
    """Isolate and center a digit at the checkpoint's expected resolution."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    # Preserve already prepared native grayscale inputs and legacy examples.
    if image.ndim == 2 and image.shape == (image_size, image_size):
        grayscale = image
        if invert == "yes" or (invert == "auto" and _border_mean(image) > image.mean()):
            grayscale = 255 - image
        if int(grayscale.max()) - int(grayscale.min()) < 8:
            raise ValueError("No foreground digit was detected in the image.")
    else:
        grayscale = normalize_digit(extract_digit(image, invert), image_size)
    normalized = grayscale.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor, grayscale


def preprocess_number_image(
    path: Path, invert: str = "auto", image_size: int = 28,
) -> tuple[torch.Tensor, np.ndarray, tuple[tuple[int, int, int, int], ...]]:
    """Build a batch of digit crops and an ordered preview strip."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    regions = extract_digits(image, invert)
    prepared = [normalize_digit(region.foreground, image_size) for region in regions]
    batch = np.stack(prepared).astype(np.float32) / 255
    tensor = torch.from_numpy(batch).unsqueeze(1)
    preview = np.concatenate(prepared, axis=1)
    return tensor, preview, tuple(region.box for region in regions)


def load_digit_model(checkpoint_path: Path) -> tuple[DigitCNN, list[int]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    image_size = checkpoint["image_shape"][-1]
    model = DigitCNN(num_classes=len(checkpoint["classes"]), image_size=image_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["classes"]


@dataclass(frozen=True)
class NumberPrediction:
    text: str
    digits: tuple[int, ...]
    scores: tuple[float, ...]
    boxes: tuple[tuple[int, int, int, int], ...]
    preprocessed: np.ndarray


def predict_number(
    path: Path, checkpoint_path: Path = PROJECT_DIR / "artifacts/best_model.pt",
    invert: str = "auto",
) -> NumberPrediction:
    """Recognize separated digits as text, preserving leading zeros."""
    model, classes = load_digit_model(checkpoint_path)
    tensor, preview, boxes = preprocess_number_image(path, invert, model.image_size)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)
    scores, indices = probabilities.max(dim=1)
    digits = tuple(classes[int(index)] for index in indices)
    return NumberPrediction(
        "".join(str(digit) for digit in digits), digits,
        tuple(float(score) for score in scores), boxes, preview,
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    if args.number:
        result = predict_number(args.image, args.checkpoint, args.invert)
        print(f"Prediction: {result.text}")
        print("Digits left to right (uncalibrated model scores):")
        for position, (digit, score) in enumerate(zip(result.digits, result.scores, strict=True), 1):
            print(f"  {position}: {digit} ({score:.1%})")
        preprocessed = result.preprocessed
    else:
        model, classes = load_digit_model(args.checkpoint)
        image_tensor, preprocessed = preprocess_image(args.image, args.invert, model.image_size)
        with torch.inference_mode():
            probabilities = model(image_tensor).softmax(dim=1)[0]
        top_k = min(max(args.top_k, 1), len(classes))
        scores, indices = probabilities.topk(top_k)
        print(f"Prediction: {classes[int(indices[0])]}")
        print("Top model scores (uncalibrated):")
        for score, index in zip(scores, indices, strict=True):
            print(f"  {classes[int(index)]}: {float(score):.1%}")

    if args.save_preprocessed:
        args.save_preprocessed.parent.mkdir(parents=True, exist_ok=True)
        preview_width = round(256 * preprocessed.shape[1] / preprocessed.shape[0])
        preview = cv2.resize(preprocessed, (preview_width, 256), interpolation=cv2.INTER_NEAREST)
        if not cv2.imwrite(str(args.save_preprocessed), preview):
            raise OSError(f"Could not save preview: {args.save_preprocessed}")
        print(f"Preprocessed preview: {args.save_preprocessed}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Error: {error}") from None
