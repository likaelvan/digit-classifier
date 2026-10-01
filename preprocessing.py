"""Isolate digits and normalize each to a centered grayscale canvas."""

from dataclasses import dataclass

import cv2
import numpy as np


def normalize_digit(foreground: np.ndarray, image_size: int = 28) -> np.ndarray:
    """Fit the strokes in a 20/28-size box and center their center of mass."""
    points = cv2.findNonZero((foreground > 0).astype(np.uint8))
    if points is None:
        raise ValueError("No foreground digit was detected in the image.")
    x, y, width, height = cv2.boundingRect(points)
    crop = foreground[y:y + height, x:x + width]
    extent = max(1, round(image_size * 20 / 28))
    scale = extent / max(width, height)
    resized = cv2.resize(
        crop, (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    top = (image_size - resized.shape[0]) // 2
    left = (image_size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    moments = cv2.moments(canvas)
    center = (image_size - 1) / 2
    dx = center - moments["m10"] / moments["m00"]
    dy = center - moments["m01"] / moments["m00"]
    return cv2.warpAffine(
        canvas, np.float32([[1, 0, dx], [0, 1, dy]]),
        (image_size, image_size), flags=cv2.INTER_LINEAR,
    )


def _color_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        return np.rint(image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return image[:, :, :3]


def _foreground_masks(
    image: np.ndarray, invert: str, max_side: int = 768,
) -> tuple[np.ndarray, list[np.ndarray], float]:
    if invert not in {"auto", "yes", "no"}:
        raise ValueError("invert must be auto, yes, or no.")
    color = _color_image(image)
    # Bound connected-component work for large photographs.
    scale = min(1.0, max_side / max(color.shape[:2]))
    if scale < 1:
        color = cv2.resize(color, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if int(gray.max()) - int(gray.min()) < 8:
        raise ValueError("No foreground digit was detected in the image.")
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = []
    if invert in {"auto", "no"}:
        masks.append(bright)
    if invert in {"auto", "yes"}:
        masks.append(255 - bright)
    if invert == "auto":
        border = np.concatenate((color[0], color[-1], color[:, 0], color[:, -1]))
        background = np.median(border, axis=0)
        distance = np.linalg.norm(color.astype(np.float32) - background, axis=2)
        distance = np.clip(distance / np.sqrt(3), 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(mask)
    return gray, masks, scale


def _fill_small_holes(foreground: np.ndarray) -> np.ndarray:
    """Fill texture-sized holes while preserving the large loops in digits."""
    contours, hierarchy = cv2.findContours(foreground, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hole_limit = float((foreground > 0).sum()) * 0.015
        for index, contour in enumerate(contours):
            if hierarchy[0, index, 3] >= 0 and cv2.contourArea(contour) < hole_limit:
                cv2.drawContours(foreground, [contour], -1, 255, cv2.FILLED)
    return foreground


def extract_digit(image: np.ndarray, invert: str = "auto") -> np.ndarray:
    """Select a substantial stroke component, excluding borders and filled badges.

    Both polarities allow a light digit inside a dark/colored badge. A color
    distance mask preserves colored/textured strokes. Selection uses geometry,
    never the predicted class or filename.
    """
    gray, masks, _ = _foreground_masks(image, invert)

    height, width = gray.shape
    candidates = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, components, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for component in range(1, count):
            x, y, w, h, area = map(int, stats[component])
            if area < max(8, height * width * 0.0005) or h < height * 0.12:
                continue
            # A tightly cropped digit may touch edges; a surrounding background
            # usually connects multiple corners. Reject it without losing strokes.
            corners = (components[0, 0], components[0, -1], components[-1, 0], components[-1, -1])
            if sum(value == component for value in corners) >= 2:
                continue
            if w / h > 1.5:
                continue
            occupancy = area / (w * h)
            if occupancy > 0.68 and w / h > 0.7:
                continue  # A filled panel, rather than strokes.
            cx, cy = centroids[component]
            off_center = np.hypot((cx - width / 2) / width, (cy - height / 2) / height)
            score = area * (1 - min(off_center, 0.8))
            selected = components == component
            # Broken handwriting can have disconnected strokes inside its box.
            for other in range(1, count):
                ox, oy, ow, oh, other_area = map(int, stats[other])
                if other != component and other_area >= area * 0.02:
                    if x <= ox and y <= oy and ox + ow <= x + w and oy + oh <= y + h:
                        selected |= components == other
            candidates.append((score, selected))
    if not candidates:
        raise ValueError("No isolated digit was detected; use a margin around one digit.")
    _, selected = max(candidates, key=lambda candidate: candidate[0])
    foreground = selected.astype(np.uint8) * 255
    return _fill_small_holes(foreground)


@dataclass(frozen=True)
class DigitRegion:
    foreground: np.ndarray
    box: tuple[int, int, int, int]


def _contains(outer: np.ndarray, inner: np.ndarray) -> bool:
    x, y, w, h = map(int, outer[:4])
    ix, iy, iw, ih = map(int, inner[:4])
    return x <= ix and y <= iy and ix + iw <= x + w and iy + ih <= y + h


def extract_digits(image: np.ndarray, invert: str = "auto") -> list[DigitRegion]:
    """Find a single row of separated digits, sorted from left to right.

    Components are selected using geometry and background contrast. Disconnected
    strokes contained by a digit's box are merged; unrelated small text and
    filled backgrounds are excluded. Boxes use original-image coordinates.
    """
    gray, masks, scale = _foreground_masks(image, invert, max_side=1536)
    height, width = gray.shape
    choices = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
        corners = (components[0, 0], components[0, -1], components[-1, 0], components[-1, -1])
        eligible = []
        for component in range(1, count):
            x, y, w, h, area = map(int, stats[component])
            if area < max(8, height * width * 0.00015) or h < max(5, height * 0.05):
                continue
            if sum(value == component for value in corners) >= 2:
                continue
            if area / (w * h) > 0.68 and w / h > 0.7:
                continue
            eligible.append(component)
        # A disconnected piece inside another stroke's box belongs to that digit.
        primary = [
            component for component in eligible
            if not any(
                other != component and stats[other, 4] >= stats[component, 4]
                and _contains(stats[other], stats[component])
                for other in eligible
            )
        ]
        if not primary:
            continue
        tallest = max(stats[component, 3] for component in primary)
        primary = [component for component in primary if stats[component, 3] >= tallest * 0.45]
        groups = []
        for component in primary:
            x, y, w, h, area = map(int, stats[component])
            selected = components == component
            for other in range(1, count):
                if other != component and stats[other, 4] >= area * 0.02:
                    if _contains(stats[component], stats[other]):
                        selected |= components == other
            foreground = _fill_small_holes(selected.astype(np.uint8) * 255)
            groups.append((foreground[y:y + h, x:x + w], (x, y, w, h)))
        choices.append((sum(int((crop > 0).sum()) for crop, _ in groups), groups, stats, primary))
    if not choices:
        raise ValueError("No digits were detected; use a clear image with space around the number.")
    _, groups, stats, primary = max(choices, key=lambda choice: choice[0])
    tops = [box[1] for _, box in groups]
    bottoms = [box[1] + box[3] for _, box in groups]
    if max(tops) >= min(bottoms):
        raise ValueError("Number mode supports one horizontal row of digits; crop one row first.")
    if any(box[2] / box[3] > 1.35 for _, box in groups):
        raise ValueError("Some digits may touch or overlap; use an image with separated digits.")
    # Do not silently turn common decimal/signed inputs into a different integer.
    typical_height = float(np.median([box[3] for _, box in groups]))
    line_top, line_bottom = min(tops), max(bottoms)
    line_left = min(box[0] for _, box in groups) - typical_height * 2
    line_right = max(box[0] + box[2] for _, box in groups) + typical_height * 2
    for component in range(1, len(stats)):
        if component in primary or any(_contains(stats[digit], stats[component]) for digit in primary):
            continue
        x, y, w, h, area = map(int, stats[component])
        if area < 8 or not line_left <= x + w / 2 <= line_right:
            continue
        cy = y + h / 2
        decimal = (
            h <= typical_height * 0.3 and w <= typical_height * 0.4 and 0.4 <= w / h <= 2.5
            and line_bottom - typical_height * 0.25 <= cy <= line_bottom + typical_height * 0.3
        )
        minus = (
            h <= typical_height * 0.22 and w >= typical_height * 0.25
            and line_top + typical_height * 0.2 <= cy <= line_bottom - typical_height * 0.2
        )
        if decimal or minus:
            raise ValueError("Number mode reads digits 0-9 only; decimal points and signs are not supported.")
    regions = []
    for foreground, (x, y, w, h) in sorted(groups, key=lambda group: group[1][0]):
        left, top = int(round(x / scale)), int(round(y / scale))
        right = min(image.shape[1], int(round((x + w) / scale)))
        bottom = min(image.shape[0], int(round((y + h) / scale)))
        regions.append(DigitRegion(foreground, (left, top, right - left, bottom - top)))
    return regions
