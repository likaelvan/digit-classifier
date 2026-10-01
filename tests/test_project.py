import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from data import create_dataloaders, make_printed_digits
from model import DigitCNN
from predict import preprocess_image
from train import run_epoch
from preprocessing import extract_digit, normalize_digit


class ProjectTests(unittest.TestCase):
    def test_model_output_shape(self) -> None:
        model = DigitCNN()
        output = model(torch.zeros(4, 1, 8, 8))
        self.assertEqual(output.shape, (4, 10))

    def test_higher_resolution_model_and_checkpoint_round_trip(self) -> None:
        model = DigitCNN(image_size=28).eval()
        inputs = torch.rand(2, 1, 28, 28)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = DigitCNN(image_size=28).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True))
            with torch.inference_mode():
                torch.testing.assert_close(model(inputs), restored(inputs))
                self.assertEqual(restored(inputs).shape, (2, 10))

    def test_data_splits_and_tensor_shapes(self) -> None:
        loaders = create_dataloaders(batch_size=32, seed=7)
        self.assertEqual(len(loaders.train.dataset), 1257)
        self.assertEqual(len(loaders.validation.dataset), 270)
        self.assertEqual(len(loaders.test.dataset), 270)
        images, labels = next(iter(loaders.train))
        self.assertEqual(images.shape, (32, 1, 8, 8))
        self.assertEqual(labels.shape, (32,))
        self.assertGreaterEqual(float(images.min()), 0)
        self.assertLessEqual(float(images.max()), 1)

    def test_one_training_epoch_updates_model(self) -> None:
        loaders = create_dataloaders(batch_size=128, seed=7)
        model = DigitCNN()
        before = model.features[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss, accuracy = run_epoch(
            model,
            loaders.train,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            optimizer,
        )
        self.assertTrue(np.isfinite(loss))
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)
        self.assertFalse(torch.equal(before, model.features[0].weight.detach()))

    def test_opencv_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = np.full((100, 100), 255, dtype=np.uint8)
            cv2.putText(image, "7", (22, 78), cv2.FONT_HERSHEY_SIMPLEX, 2.4, 0, 6)
            image_path = Path(directory) / "digit.png"
            cv2.imwrite(str(image_path), image)

            tensor, processed = preprocess_image(image_path, invert="auto")
            self.assertEqual(tensor.shape, (1, 1, 8, 8))
            self.assertEqual(processed.shape, (8, 8))
            self.assertGreaterEqual(float(tensor.min()), 0)
            self.assertLessEqual(float(tensor.max()), 1)

    def test_blank_images_are_rejected_at_native_and_photo_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blank.png"
            for size in (8, 28, 100):
                for value in (0, 128, 255):
                    cv2.imwrite(str(path), np.full((size, size), value, dtype=np.uint8))
                    with self.subTest(size=size, value=value):
                        with self.assertRaises(ValueError):
                            preprocess_image(path, image_size=28 if size != 8 else 8)

    def test_polarity_and_footer_do_not_change_the_digit(self) -> None:
        clean = np.full((200, 160), 255, dtype=np.uint8)
        cv2.putText(clean, "9", (45, 135), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
        expected = normalize_digit(extract_digit(clean))
        inverted = normalize_digit(extract_digit(255 - clean))
        np.testing.assert_array_equal(expected, inverted)
        footer = clean.copy()
        cv2.rectangle(footer, (0, 182), (159, 199), 0, -1)
        np.testing.assert_array_equal(expected, normalize_digit(extract_digit(footer)))

    def test_light_digit_isolated_from_colored_badge(self) -> None:
        badge = np.full((200, 200, 3), 255, dtype=np.uint8)
        cv2.circle(badge, (100, 100), 88, (0, 100, 255), -1)
        cv2.putText(badge, "1", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        strokes = normalize_digit(extract_digit(badge))
        plain = np.zeros((200, 200), dtype=np.uint8)
        cv2.putText(plain, "1", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 6)
        expected = normalize_digit(extract_digit(plain))
        self.assertLess(float(np.abs(strokes.astype(float) - expected.astype(float)).mean()), 4)

    def test_transparency_does_not_become_a_dark_background(self) -> None:
        image = np.zeros((200, 160, 4), dtype=np.uint8)
        cv2.putText(image, "9", (45, 135), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0, 255), 5)
        opaque = np.full((200, 160), 255, dtype=np.uint8)
        cv2.putText(opaque, "9", (45, 135), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 5)
        np.testing.assert_array_equal(
            normalize_digit(extract_digit(image)), normalize_digit(extract_digit(opaque)),
        )

    def test_centering_and_printed_data(self) -> None:
        source = np.zeros((80, 80), dtype=np.uint8)
        source[20:70, 15:22] = 255
        centered = normalize_digit(source)
        moments = cv2.moments(centered)
        self.assertAlmostEqual(moments["m10"] / moments["m00"], 13.5, delta=0.1)
        self.assertAlmostEqual(moments["m01"] / moments["m00"], 13.5, delta=0.1)
        first_images, first_labels = make_printed_digits(20, 7)
        second_images, second_labels = make_printed_digits(20, 7)
        np.testing.assert_array_equal(first_images, second_images)
        np.testing.assert_array_equal(first_labels, second_labels)
        self.assertEqual(first_images.shape, (20, 28, 28))
        self.assertEqual(set(first_labels.tolist()), set(range(10)))


if __name__ == "__main__":
    unittest.main()
