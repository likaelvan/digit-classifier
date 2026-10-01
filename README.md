# CNN Handwritten-Digit Practice

This project trains a PyTorch convolutional neural network to recognize digits from 0 through 9. OpenCV isolates the digits in a new image before the CNN makes its predictions. The default mode reads one digit; `--number` reads one horizontal row of separated digits.

**Testing instructions and accuracy metrics updated: 1 October 2026.**

## Current Accuracy

These results use the saved model at `artifacts/best_model.pt`:

| Mode / test set | Correct / total | Accuracy |
|---|---|---|
| Single digits: held-out MNIST test images | 9,923 / 10,000 | **99.23%** |
| Single digits: supplied workspace images | 4 / 4 | **100%** |
| Numbers: complete digit strings in supplied examples | 6 / 6 | **100% exact match** |
| Individual digits within those number examples | 26 / 26 | **100%** |

A number counts as correct only when every digit, its order, and any leading zeros match the expected string. The four digit images and six number images are small regression sets. Number samples are generated printed glyphs or assembled handwritten digits; accuracy on natural multi-digit handwriting has not yet been measured. See **Evaluation** below for the saved reports and commands.

## Setup After Cloning

Run these commands from the cloned project directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python predict.py test_inputs/digit_6_handwritten.jpg
python predict.py test_inputs/numbers/number_123.png --number
```

The repository includes the trained checkpoint, the baseline checkpoint, accuracy reports, and test inputs. Inference works immediately after installing dependencies; training is optional. The MNIST download cache, additional training runs, generated previews, and local configuration are excluded by `.gitignore`.

## Quick Testing in Jupyter

Open `README.md` in the project folder using Jupyter's file browser. If it was already open, close its tab and reopen it to load the version saved on disk. `.ipynb_checkpoints/README-checkpoint.md` is an older automatic backup.

Open a separate Jupyter terminal tab for the Python commands below. Run them from the project folder, the directory containing `README.md` and `predict.py`.

### Check the Four Supplied Images

```bash
python evaluate_workspace.py
```

Expected final line:

```text
Workspace regression: 4/4 correct
```

The input images are in `test_inputs/`. The evaluation reads their expected digits from `test_inputs/labels.json` and writes its report to `artifacts/workspace_evaluation/results.json`. Preprocessing previews are saved in the same output folder.

### Predict One Supplied Image

```bash
python predict.py test_inputs/digit_6_handwritten.jpg
```

Expected prediction: `6`. To save the exact processed image for inspection:

```bash
python predict.py test_inputs/digit_6_handwritten.jpg --save-preprocessed artifacts/workspace_evaluation/digit_6_preview.png
```

### Predict Your Own Image

Use Jupyter's file browser to upload your single-digit image into `test_inputs/`. Replace `my_digit.png` below with the uploaded filename, including its extension:

```bash
python predict.py test_inputs/my_digit.png
```

This uses the saved model at `artifacts/best_model.pt`. Prediction does not require retraining, an IP address argument, or a connection to a model API.

To add your image to the labeled batch evaluation, see **Test Your Own Image** below for the `labels.json` entry.

### Predict a Number with Multiple Digits

The new number mode reuses the saved CNN: it finds each digit, sorts the crops from left to right, predicts them as a batch, and joins the predictions into text. Leading zeros are retained.

```bash
python predict.py test_inputs/numbers/number_123.png --number
python predict.py test_inputs/numbers/number_007.png --number
```

Expected predictions: `123` and `007`.

For your own number image, upload it to `test_inputs/numbers/`, then replace the filename below:

```bash
python predict.py test_inputs/numbers/my_number.png --number
```

Inspect the digit crops in their reading order:

```bash
python predict.py test_inputs/numbers/number_123.png --number --save-preprocessed artifacts/number_evaluation/number_123_preview.png
```

Run the six supplied number examples:

```bash
python evaluate_workspace.py --manifest test_inputs/numbers/labels.json --output-dir artifacts/number_evaluation
```

Expected final line: `Workspace regression: 6/6 correct`. The samples include generated printed numbers, an inverted-color example, and a number assembled from held-out handwritten MNIST digits. This is a regression check, not a benchmark of natural multi-digit handwriting.

Number mode supports unsigned digit strings in one row with visible space between digits. Multiple rows and some touching-digit groups are rejected. Clear decimal points and minus signs are rejected instead of silently changing the number; other symbols and heavily connected handwriting are unsupported. Use the saved crop preview to check segmentation on a new image.

### Copy Commands from the Browser Terminal

In JupyterLab on Windows/Linux, hold **Shift** while dragging across terminal text, then release Shift and press **Ctrl+C**. JupyterLab can use **Ctrl+Shift+C** for its command palette. For the browser's context menu, try **Shift + right-click**. See the [JupyterLab terminal copy/paste instructions](https://jupyterlab.readthedocs.io/en/stable/user/terminal.html#copy-paste).

If Codex captures the mouse selection, type `/raw on` in the Codex chat to enable output intended for easier terminal selection. This is a Codex command; run the Python commands above in your shell terminal. See [official OpenAI documentation](https://learn.chatgpt.com/docs/developer-commands#toggle-raw-scrollback-with-raw).

## What Changed

The original model used 1,797 images at `8 x 8` resolution. Its preprocessing sometimes selected a watermark, colored badge, or background instead of the digit. All four workspace images failed with that version.

The default model now uses `28 x 28` inputs and trains on:

- 54,000 MNIST training images, with 6,000 separate validation images;
- 1,257 images from the original scikit-learn digits training subset, preserving its validation and test splits;
- 6,000 generated printed digits from OpenCV and available system fonts.

That is **61,257 training examples**. The installed model was trained from random weights on the CPU for **8 epochs**, using batches of 128. The downloaded archive contains image data and labels; no pretrained model weights were downloaded. Configuration and all eight epochs of learning history are recorded in `artifacts/metrics.json`.

Training adds small rotations, translations, scaling, width changes, and stroke-thickness changes. Validation and test images are never augmented. Checkpoint selection uses only MNIST validation accuracy. The four workspace images are regression examples and are never loaded by training or used for checkpoint selection.

MNIST's official 10,000-image test split stays separate. The archive comes from the [official Keras MNIST source](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py) and is checked against its SHA-256 checksum. Keras itself is not required.

## Run Inference

The improved checkpoint is already available at `artifacts/best_model.pt`:

```bash
python predict.py test_inputs/digit_6_handwritten.jpg
python predict.py test_inputs/digit_1_badge.jpg
python predict.py test_inputs/digit_9_graphic.png
python predict.py test_inputs/digit_9_photo.jpg
```

Inspect the exact image received by the CNN:

```bash
python predict.py test_inputs/digit_6_handwritten.jpg \
  --save-preprocessed artifacts/workspace_evaluation/digit_6_preview.png
```

Preprocessing handles transparency, considers both light and dark strokes, removes unrelated components such as footer text, separates a light digit from a colored badge, and centers the digit's center of mass. It preserves disconnected handwriting strokes inside the selected digit's box. Input resolution is read from the checkpoint; the original `8 x 8` checkpoint remains loadable.

Use an image containing one clear digit in the default mode, or add `--number` for a row of separated digits. Scores printed by the model are uncalibrated softmax values. They should not be interpreted as a guarantee of correctness. Blank images are rejected. Number mode prints an individual score for each digit and does not assign a confidence to the complete number.

## Reproduce Training

```bash
python train.py --device cpu --dataset mnist --epochs 8 \
  --output-dir artifacts/new_run
```

The first MNIST run downloads about 11 MB to `datasets/mnist.npz`. Later runs reuse the verified cache. PyTorch defaults to two CPU threads for this project; change this with `--threads`. Printed examples use available DejaVu/URW fonts and fall back to OpenCV fonts, so installing a different font set can change that part of the training data.

Every output directory receives:

- `best_model.pt`, containing weights, class labels, image shape, and run configuration;
- `metrics.json`, with configuration, package versions, validation/test metrics, and training history;
- `learning_curves.png` and `confusion_matrix.png`;
- `example_digit_<label>.png`, from the held-out test dataset.

Choose a different `--output-dir` for each experiment. Reusing a directory overwrites its model and reports. The original model and artifacts are preserved in `artifacts/baseline/`.

For the smaller offline exercise:

```bash
python train.py --dataset digits --device cpu --epochs 20 \
  --output-dir artifacts/offline_digits
```

This mode uses the original `8 x 8` CNN and the bundled dataset without a download. It is useful for learning the training loop, but has limited coverage of external image styles.

## Evaluation

The digit model was trained on 30 September 2026. The current digit and number regression results were checked on 1 October 2026; number mode reuses the same weights without retraining.

| Evaluation | Correct / total | Accuracy | Additional metric / comparison |
|---|---|---|---|
| MNIST validation | 5,928 / 6,000 | 98.80% | Used to select the checkpoint |
| MNIST single-digit test | 9,923 / 10,000 | 99.23% | Macro precision 0.9923; macro recall 0.9922; macro F1 0.9922 |
| Original scikit-learn digits test | 260 / 270 | 96.30% | Original checkpoint: 96.67% |
| Workspace single-digit regression | 4 / 4 | 100% | Original application: 0 / 4 |
| Number regression: complete strings | 6 / 6 | 100% exact match | Includes `007`, inverted colors, and assembled handwriting |
| Number regression: individual digits | 26 / 26 | 100% | Digits compared at the same position in the expected string |

Saved evidence: [training and held-out digit metrics](artifacts/metrics.json), [workspace digit results](artifacts/workspace_evaluation/results.json), and [number results](artifacts/number_evaluation/results.json). The 26-digit result is calculated from the expected and predicted strings in the number report.

These datasets measure different input styles. MNIST test accuracy measures single-digit classification; it does not measure number segmentation or complete-number accuracy. The workspace regression sets were used while developing preprocessing and do not establish accuracy on arbitrary photos, graphics, or natural multi-digit handwriting. Model scores printed during prediction are separate from these measured accuracy results.

Run that regression check, including saved preprocessing previews:

```bash
python evaluate_workspace.py
```

Run the number regression check separately:

```bash
python evaluate_workspace.py --manifest test_inputs/numbers/labels.json --output-dir artifacts/number_evaluation
```

Expected results are `4/4 correct` for single digits and `6/6 correct` for complete numbers. Upload and label more of your own images to expand these checks; rerunning evaluation updates the JSON reports. Refresh the README metrics when the labeled test sets change.

Test inputs and their visually assigned labels are grouped together:

```text
test_inputs/
    digit_1_badge.jpg
    digit_6_handwritten.jpg
    digit_9_graphic.png
    digit_9_photo.jpg
    labels.json
    numbers/
        number_123.png
        number_007.png
        number_908.png
        number_1234567890.png
        number_123_inverted.png
        number_2468_handwritten.png
        labels.json
```

`python evaluate_workspace.py` checks every image listed in `test_inputs/labels.json` against its expected digit. Results and previews are saved under `artifacts/workspace_evaluation/`; inputs are kept separate from generated outputs. The current expected result is `Workspace regression: 4/4 correct`.

### Test Your Own Image

Save a single-digit image in `test_inputs/`, then run from the project directory:

```bash
python predict.py test_inputs/my_digit.png
```

To inspect the processed input:

```bash
python predict.py test_inputs/my_digit.png \
  --save-preprocessed artifacts/workspace_evaluation/my_digit_preview.png
```

For a batch accuracy check, also add your image and its expected label to the `images` list in `test_inputs/labels.json`:

```json
{"path": "my_digit.png", "label": 5}
```

Use the actual digit as `label`, an integer from 0 to 9. Paths are relative to the folder containing `labels.json`. Then rerun `python evaluate_workspace.py`. You can choose a separate output folder with `--output-dir artifacts/my_test_results`.

For number images, add an entry to `test_inputs/numbers/labels.json` with `mode` set to `number` and the expected value written as a string:

```json
{"path": "my_number.png", "label": "007", "mode": "number"}
```

The string label preserves leading zeros. Evaluate this manifest with the number-example command in **Predict a Number with Multiple Digits** above. Number accuracy counts a sample as correct only when the complete digit string matches.

Run the nineteen automated tests:

```bash
python -m unittest discover -s tests -v
```

Tests cover model/checkpoint compatibility, data splits, training updates, blank input rejection, centering, transparency, polarity, footer removal, colored-badge isolation, digit ordering, leading zeros, image coordinates, disconnected strokes, and rejection of multiple rows and common unsupported symbols.

## Project Files

| File | Purpose |
|---|---|
| `model.py` | CNN supporting legacy 8-pixel and new 28-pixel inputs. |
| `preprocessing.py` | Single-digit isolation, multi-digit segmentation, and shared sizing/centering. |
| `data.py` | Dataset download, checksum validation, splits, and synthetic glyph generation. |
| `train.py` | Augmentation, training, validation selection, and held-out evaluation. |
| `predict.py` | Single-digit inference, `predict_number()` / `--number`, and optional preprocessing preview. |
| `evaluate_workspace.py` | Separate evaluation of labeled workspace images. |
| `test_inputs/` | User test images and `labels.json` with expected digits. |
| `tests/test_project.py` | Behavioral regression tests. |
| `tests/test_numbers.py` | Multi-digit segmentation and complete-number regression tests. |

## Understand the CNN

```text
1 x 28 x 28 grayscale image
    -> convolution: 16 x 28 x 28
    -> convolution: 16 x 28 x 28
    -> max pooling: 16 x 14 x 14
    -> convolution: 32 x 14 x 14
    -> max pooling: 32 x 7 x 7
    -> flatten: 1,568 values
    -> linear layer: 64 values
    -> output: 10 class scores
```

The CNN learns its filters and classifier from labeled examples. OpenCV prepares the pixels. Cross-entropy measures training loss, AdamW updates the weights, and validation accuracy selects the saved model. Test accuracy and per-class precision, recall, F1, and the confusion matrix assess the selected model.
