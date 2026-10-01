"""Behavioral checks for segmentation and complete-number recognition."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from predict import predict_number, preprocess_number_image
from preprocessing import extract_digits, normalize_digit


def render_number(text: str) -> np.ndarray:
    image = np.full((150, 110 * len(text) + 40), 255, dtype=np.uint8)
    for index, digit in enumerate(text):
        cv2.putText(
            image, digit, (25 + 110 * index, 115),
            cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5, cv2.LINE_AA,
        )
    return image


class NumberTests(unittest.TestCase):
    def test_all_regions_are_sorted_and_narrow_digits_are_retained(self) -> None:
        regions = extract_digits(render_number("1234567890"))
        self.assertEqual(len(regions), 10)
        lefts = [region.box[0] for region in regions]
        self.assertEqual(lefts, sorted(lefts))
        self.assertLess(regions[0].box[2], regions[1].box[2])

    def test_polarity_footer_and_transparency_preserve_digit_crops(self) -> None:
        image = render_number("007")
        expected = [normalize_digit(region.foreground) for region in extract_digits(image)]
        footer = image.copy()
        cv2.rectangle(footer, (0, 135), (footer.shape[1] - 1, 149), 0, -1)
        transparent = np.zeros((*image.shape, 4), dtype=np.uint8)
        transparent[:, :, 3] = 255 - image
        for variant in (255 - image, footer, transparent):
            regions = extract_digits(variant)
            self.assertEqual(len(regions), 3)
            for wanted, actual in zip(expected, regions, strict=True):
                # Footer pixels can slightly change Otsu's threshold at soft edges.
                prepared = normalize_digit(actual.foreground)
                self.assertLess(float(np.abs(wanted.astype(float) - prepared.astype(float)).mean()), 2)
                self.assertGreater(float(((wanted > 0) == (prepared > 0)).mean()), 0.98)

    def test_decimal_and_minus_are_not_silently_discarded(self) -> None:
        for text in ("12.5", "-123"):
            with self.subTest(number=text):
                with self.assertRaisesRegex(ValueError, "decimal points and signs"):
                    extract_digits(render_number(text))

    def test_two_rows_are_rejected(self) -> None:
        row = render_number("123")
        with self.assertRaisesRegex(ValueError, "one horizontal row"):
            extract_digits(np.concatenate((row, row), axis=0))

    def test_blank_number_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No foreground"):
            extract_digits(np.full((150, 400), 255, dtype=np.uint8))

    def test_fragment_inside_a_digit_is_merged(self) -> None:
        image = np.full((150, 240), 255, dtype=np.uint8)
        cv2.line(image, (50, 20), (50, 110), 0, 5)
        cv2.line(image, (50, 110), (100, 110), 0, 5)
        cv2.line(image, (100, 110), (100, 75), 0, 5)
        cv2.line(image, (70, 80), (86, 80), 0, 5)
        cv2.putText(image, "1", (160, 112), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
        regions = extract_digits(image)
        self.assertEqual(len(regions), 2)
        first = regions[0]
        self.assertGreater(int(first.foreground[80 - first.box[1], 76 - first.box[0]]), 0)

    def test_box_coordinates_match_the_original_large_image(self) -> None:
        image = cv2.resize(render_number("123"), None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        regions = extract_digits(image)
        self.assertEqual(len(regions), 3)
        for index, region in enumerate(regions):
            x, y, w, h = region.box
            self.assertGreater(x, 110 * index * 8)
            self.assertLess(x + w, (110 * (index + 1) + 25) * 8)
            self.assertGreater(h, 60 * 8)
            self.assertGreater(y, 30 * 8)

    def test_number_batch_and_preview_preserve_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "123.png"
            cv2.imwrite(str(path), render_number("123"))
            tensor, preview, boxes = preprocess_number_image(path)
            self.assertEqual(tensor.shape, (3, 1, 28, 28))
            self.assertEqual(preview.shape, (28, 84))
            self.assertEqual(len(boxes), 3)
            for index in range(3):
                np.testing.assert_allclose(
                    tensor[index, 0].numpy(), preview[:, index * 28:(index + 1) * 28] / 255,
                    atol=1e-7,
                )

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[1] / "artifacts/best_model.pt").is_file(),
        "Trained checkpoint is not available.",
    )
    def test_saved_model_reads_numbers_and_preserves_leading_zeros(self) -> None:
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            for text in ("123", "007", "908", "1234567890"):
                path = Path(directory) / f"{text}.png"
                cv2.imwrite(str(path), render_number(text))
                with self.subTest(number=text):
                    result = predict_number(path)
                    self.assertEqual(result.text, text)
                    self.assertEqual(len(result.digits), len(text))


if __name__ == "__main__":
    unittest.main()
