"""Evaluate visually labeled workspace images separately from model training."""

import argparse
import json
from pathlib import Path

import cv2
import torch

from model import DigitCNN
from predict import preprocess_image, preprocess_number_image


PROJECT_DIR = Path(__file__).resolve().parent


def evaluate(checkpoint_path: Path, manifest_path: Path, output_dir: Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    image_size = checkpoint["image_shape"][-1]
    classes = checkpoint["classes"]
    model = DigitCNN(num_classes=len(classes), image_size=image_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    manifest = json.loads(manifest_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for example in manifest["images"]:
        image_path = manifest_path.parent / example["path"]
        mode = example.get("mode", "digit")
        if mode == "number":
            if not isinstance(example["label"], str) or not example["label"].isascii() or not example["label"].isdigit():
                raise ValueError("Number labels must be digit strings, for example \"007\".")
            tensor, processed, _ = preprocess_number_image(image_path, image_size=image_size)
        elif mode == "digit":
            tensor, processed = preprocess_image(image_path, image_size=image_size)
        else:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        with torch.inference_mode():
            probabilities = model(tensor).softmax(dim=1)
        scores, indices = probabilities.max(dim=1)
        prediction = (
            "".join(str(classes[int(index)]) for index in indices)
            if mode == "number" else classes[int(indices[0])]
        )
        preview_path = output_dir / f"{Path(example['path']).stem}_preprocessed.png"
        preview_width = round(280 * processed.shape[1] / processed.shape[0])
        preview = cv2.resize(processed, (preview_width, 280), interpolation=cv2.INTER_NEAREST)
        if not cv2.imwrite(str(preview_path), preview):
            raise OSError(f"Could not save preview: {preview_path}")
        result = {
            "image": example["path"], "expected": example["label"], "predicted": prediction,
            "mode": mode,
            "correct": prediction == example["label"],
        }
        if mode == "number":
            result["digit_scores_uncalibrated"] = [float(score) for score in scores]
        else:
            result["softmax_score_uncalibrated"] = float(scores[0])
        results.append(result)
        print(f"{result['image']}: expected {result['expected']}, predicted {prediction}")
    correct = sum(result["correct"] for result in results)
    report = {
        "description": manifest["description"], "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "image_size": image_size, "correct": correct, "total": len(results),
        "accuracy": correct / len(results) if results else None, "results": results,
    }
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Workspace regression: {correct}/{len(results)} correct")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "artifacts/best_model.pt")
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "test_inputs/labels.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "artifacts/workspace_evaluation")
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.checkpoint, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
