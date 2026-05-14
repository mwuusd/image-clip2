from __future__ import annotations

"""Border detection utilities used by the CLI.

The detector combines multiple signals (edges + threshold masks + contour scoring)
and then applies controlled expansion steps so the final rectangle tends to match
the outer phone boundary rather than inner screen details.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
np.random.seed(42)

# Canny/edge preprocessing parameters.
EDGE_BLUR_KERNEL = (7, 7)
EDGE_CANNY_SIGMA = 0.33
EDGE_MORPH_KERNEL_SIZE = 5
EDGE_DILATE_ITERATIONS = 2
EDGE_CLOSE_ITERATIONS = 3

# Threshold-mask preprocessing parameters.
MASK_BLUR_KERNEL = (5, 5)
MASK_MORPH_KERNEL_SIZE = 5
MASK_CLOSE_ITERATIONS = 2
MASK_OPEN_ITERATIONS = 1

# Contour area acceptance range, expressed as ratio of full image area.
MIN_CONTOUR_AREA_RATIO = 0.005
MAX_CONTOUR_AREA_RATIO = 0.95

# Aspect-ratio scoring windows (ratio is always >= 1.0 after normalization).
ASPECT_GOOD_MIN = 1.2
ASPECT_GOOD_MAX = 2.6
ASPECT_OK_MIN = 1.0
ASPECT_OK_MAX = 3.2
ASPECT_GOOD_PENALTY = 1.0
ASPECT_OK_PENALTY = 0.7
ASPECT_BAD_PENALTY = 0.35

# Expansion from edge evidence around the current rectangle.
EDGE_EXPAND_PAD_RATIO = 0.12
EDGE_EXPAND_MIN_PAD = 10
EDGE_EXPAND_MAX_AREA_RATIO = 0.95
EDGE_BBOX_MIN_PIXELS = 12
EDGE_BBOX_ROW_RATIO = 0.003
EDGE_BBOX_COL_RATIO = 0.005

# Guardrails for replacing suspiciously wide rectangles.
WIDE_RECT_EDGE_TOUCH_MARGIN = 1
WIDE_RECT_ONE_SIDE_RATIO = 0.85
WIDE_RECT_BOTH_SIDES_RATIO = 0.9
WIDE_RECT_RESET_MIN_SHRINK_RATIO = 0.05

# Horizontal-only expansion from threshold mask occupancy.
HORIZONTAL_EXPAND_PAD_Y_RATIO = 0.03
HORIZONTAL_EXPAND_MIN_PAD_Y = 10
HORIZONTAL_EXPAND_MIN_PIXELS = 30
HORIZONTAL_EXPAND_PIXEL_RATIO = 0.02
HORIZONTAL_EXPAND_MAX_WIDTH_RATIO = 0.95

# Soft-threshold fallback controls when the primary rectangle looks too small.
SOFT_BBOX_THRESHOLD_DELTA = 3
SOFT_FALLBACK_WIDTH_GAP_RATIO = 0.06
SOFT_FALLBACK_HEIGHT_GAP_RATIO = 0.03

# Minimum directional movement required when merging with fallback candidates.
CANDIDATE_MIN_DELTA_PIXELS = 10
CANDIDATE_MIN_DELTA_RATIO = 0.001
CANDIDATE_MAX_AREA_RATIO = 0.95

# Bottom-boundary recovery parameters for difficult images.
BOTTOM_TRIGGER_MIN_PIXELS = 120
BOTTOM_TRIGGER_WIDTH_RATIO = 0.03
BOTTOM_SOFT_THRESHOLD_DELTA = 20
BOTTOM_PAD_X_MIN = 5
BOTTOM_PAD_X_RATIO = 0.02
BOTTOM_ROW_MIN_PIXELS = 12
BOTTOM_ROW_MIN_RATIO = 0.003
BOTTOM_ANCHOR_SEARCH_RADIUS = 6
BOTTOM_MIN_EXTENSION_PIXELS = 8
BOTTOM_SPARSE_MIN_PIXELS = 5
BOTTOM_SPARSE_RATIO = 0.0015
BOTTOM_MAX_GAP_PIXELS = 300
BOTTOM_MIN_RUN_LENGTH = 20
BOTTOM_PEAK_MIN_PIXELS = 40
BOTTOM_PEAK_WIDTH_RATIO = 0.01
BOTTOM_MAX_EXTRA_MIN_PIXELS = 80
BOTTOM_MAX_EXTRA_HEIGHT_RATIO = 0.2

# Top-boundary recovery parameters for cases with weak upper contrast.
TOP_MIN_EXTENSION_PIXELS = 80
TOP_MAX_EXTRA_MIN_PIXELS = 80
TOP_MAX_EXTRA_HEIGHT_RATIO = 0.2


@dataclass(frozen=True)
class BorderRect:
    """Axis-aligned rectangle represented as top-left plus size."""

    x: int
    y: int
    w: int
    h: int

    def as_line(self) -> str:
        """Return rectangle in output text format: `x y w h`."""

        return f"{self.x} {self.y} {self.w} {self.h}"


def _debug_dir(output_dir: Path) -> Path:
    """Return `<output_dir>/debug`, creating it if needed."""

    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


def _save_debug_images(
    image: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    debug_dir: Path,
    image_name: str,
) -> None:
    """Save gray/edge/mask debug artifacts for one input image."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(debug_dir / f"{image_name}_gray.jpg"), gray)
    cv2.imwrite(str(debug_dir / f"{image_name}_edges.jpg"), edges)
    cv2.imwrite(str(debug_dir / f"{image_name}_mask.jpg"), mask)


def _load_image(image_path: Path) -> np.ndarray:
    """Load image from disk and raise a clear error when unavailable."""

    if not image_path.exists() or not image_path.is_file():
        msg = f"Image path does not exist or is not a file: {image_path}"
        raise FileNotFoundError(msg)
    image = cv2.imread(str(image_path))
    if image is None:
        msg = f"Failed to read image: {image_path}"
        raise FileNotFoundError(msg)
    return image


def _prepare_edges(gray: np.ndarray) -> np.ndarray:
    """Generate an edge map tuned for large rectangular phone borders."""

    # Smooth noise before adaptive Canny thresholding.
    blurred = cv2.GaussianBlur(gray, EDGE_BLUR_KERNEL, 0)
    median = float(np.median(blurred))
    lower = int(max(0.0, (1.0 - EDGE_CANNY_SIGMA) * median))
    upper = int(min(255.0, (1.0 + EDGE_CANNY_SIGMA) * median))
    edges = cv2.Canny(blurred, lower, upper)

    # Connect nearby fragments to make continuous border candidates.
    kernel = np.ones((EDGE_MORPH_KERNEL_SIZE, EDGE_MORPH_KERNEL_SIZE), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=EDGE_DILATE_ITERATIONS)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=EDGE_CLOSE_ITERATIONS)
    return edges


def _prepare_masks(gray: np.ndarray) -> list[np.ndarray]:
    """Return binary masks (normal + inverted) from Otsu thresholding."""

    blurred = cv2.GaussianBlur(gray, MASK_BLUR_KERNEL, 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)
    return [thresh, inverted]


def _contours_from_binary(binary: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """Clean a binary image and extract external contours from it."""

    kernel = np.ones((MASK_MORPH_KERNEL_SIZE, MASK_MORPH_KERNEL_SIZE), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=MASK_CLOSE_ITERATIONS)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=MASK_OPEN_ITERATIONS)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, cleaned


def _score_contour(contour: np.ndarray, image_shape: tuple[int, int]) -> float:
    """Compute quality score for a contour candidate.

    Higher score means the contour is more likely to represent the phone border.
    The score favors large rectangular, reasonably solid shapes.
    """

    area = float(cv2.contourArea(contour))
    if area <= 0:
        return 0.0

    # `rect_area` captures how much image area this contour could cover.
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = float(w * h)
    if rect_area <= 0:
        return 0.0
    rectangularity = area / rect_area

    # `solidity` penalizes very hollow or fragmented shapes.
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Normalize orientation by forcing ratio >= 1 (portrait/landscape both valid).
    aspect_ratio = max(float(w) / h, float(h) / w) if h > 0 and w > 0 else 0.0
    if ASPECT_GOOD_MIN <= aspect_ratio <= ASPECT_GOOD_MAX:
        aspect_penalty = ASPECT_GOOD_PENALTY
    elif ASPECT_OK_MIN <= aspect_ratio <= ASPECT_OK_MAX:
        aspect_penalty = ASPECT_OK_PENALTY
    else:
        aspect_penalty = ASPECT_BAD_PENALTY

    quality = (0.6 + 0.4 * rectangularity) * max(solidity, 0.6) * aspect_penalty
    return rect_area * quality


def _best_contour(contours: Iterable[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray | None:
    """Select the best contour after area filtering and quality scoring."""

    image_area = float(image_shape[0] * image_shape[1])
    # Reject tiny specks and near-full-image masks before expensive scoring.
    min_area = image_area * MIN_CONTOUR_AREA_RATIO
    max_area = image_area * MAX_CONTOUR_AREA_RATIO
    best = None
    best_score = 0.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue
        score = _score_contour(contour, image_shape)
        if score > best_score:
            best_score = score
            best = contour
    return best


def _bbox_from_nonzero(mask: np.ndarray) -> BorderRect | None:
    """Return bbox of non-zero pixels in a mask, or `None` if empty."""

    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    return BorderRect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def _bbox_from_edge_density(edges: np.ndarray) -> BorderRect | None:
    """Return robust edge bbox using row/column occupancy thresholds."""

    height, width = edges.shape
    rows = (edges > 0).sum(axis=1)
    cols = (edges > 0).sum(axis=0)
    min_row_pixels = max(EDGE_BBOX_MIN_PIXELS, int(width * EDGE_BBOX_ROW_RATIO))
    min_col_pixels = max(EDGE_BBOX_MIN_PIXELS, int(height * EDGE_BBOX_COL_RATIO))

    valid_rows = np.where(rows >= min_row_pixels)[0]
    valid_cols = np.where(cols >= min_col_pixels)[0]
    if valid_rows.size == 0 or valid_cols.size == 0:
        return None

    y_min = int(valid_rows.min())
    y_max = int(valid_rows.max())
    x_min = int(valid_cols.min())
    x_max = int(valid_cols.max())
    return BorderRect(x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def _touches_horizontal_edges(rect: BorderRect, image_width: int) -> bool:
    """Return True if rectangle touches left or right image edge."""

    left_touch = rect.x <= WIDE_RECT_EDGE_TOUCH_MARGIN
    right_touch = rect.x + rect.w >= image_width - WIDE_RECT_EDGE_TOUCH_MARGIN
    return left_touch or right_touch


def _is_suspiciously_wide(rect: BorderRect, image_shape: tuple[int, int]) -> bool:
    """Detect rectangles likely over-expanded to image sides by noise."""

    image_height, image_width = image_shape
    del image_height
    left_touch = rect.x <= WIDE_RECT_EDGE_TOUCH_MARGIN
    right_touch = rect.x + rect.w >= image_width - WIDE_RECT_EDGE_TOUCH_MARGIN
    width_ratio = rect.w / image_width if image_width > 0 else 0.0
    if left_touch and right_touch:
        return width_ratio >= WIDE_RECT_BOTH_SIDES_RATIO
    if left_touch or right_touch:
        return width_ratio >= WIDE_RECT_ONE_SIDE_RATIO
    return False


def _expand_rect_with_edges(rect: BorderRect, edges: np.ndarray) -> BorderRect:
    """Expand rectangle using nearby edge evidence.

    This step helps recover outer phone borders when contour extraction picks a
    slightly inner rectangle.
    """

    height, width = edges.shape
    # Search region around current rectangle (`pad_x`/`pad_y`).
    pad_x = max(EDGE_EXPAND_MIN_PAD, int(rect.w * EDGE_EXPAND_PAD_RATIO))
    pad_y = max(EDGE_EXPAND_MIN_PAD, int(rect.h * EDGE_EXPAND_PAD_RATIO))

    x0 = max(0, rect.x - pad_x)
    y0 = max(0, rect.y - pad_y)
    x1 = min(width, rect.x + rect.w + pad_x)
    y1 = min(height, rect.y + rect.h + pad_y)
    roi = edges[y0:y1, x0:x1]

    edge_bbox = _bbox_from_edge_density(roi)
    if edge_bbox is None:
        edge_bbox = _bbox_from_nonzero(roi)
    if edge_bbox is None:
        return rect

    ex0 = x0 + edge_bbox.x
    ey0 = y0 + edge_bbox.y
    ex1 = ex0 + edge_bbox.w
    ey1 = ey0 + edge_bbox.h

    nx0 = min(rect.x, ex0)
    ny0 = min(rect.y, ey0)
    nx1 = max(rect.x + rect.w, ex1)
    ny1 = max(rect.y + rect.h, ey1)
    expanded = BorderRect(nx0, ny0, nx1 - nx0, ny1 - ny0)

    # Guardrail: do not accept implausibly huge expansion.
    image_area = float(width * height)
    if expanded.w * expanded.h >= image_area * EDGE_EXPAND_MAX_AREA_RATIO:
        return rect
    if expanded.w * expanded.h <= rect.w * rect.h:
        return rect
    return expanded


def _expand_rect_horizontally_with_mask(rect: BorderRect, mask: np.ndarray) -> BorderRect:
    """Expand only left/right bounds based on mask column occupancy."""

    height, width = mask.shape
    pad_y = max(HORIZONTAL_EXPAND_MIN_PAD_Y, int(rect.h * HORIZONTAL_EXPAND_PAD_Y_RATIO))
    y0 = max(0, rect.y - pad_y)
    y1 = min(height, rect.y + rect.h + pad_y)
    roi = mask[y0:y1, :]

    if roi.size == 0:
        return rect

    # `counts` tells how many foreground pixels exist per column in ROI.
    counts = (roi > 0).sum(axis=0)
    min_pixels = max(HORIZONTAL_EXPAND_MIN_PIXELS, int(roi.shape[0] * HORIZONTAL_EXPAND_PIXEL_RATIO))
    cols = np.where(counts >= min_pixels)[0]
    if cols.size == 0:
        return rect

    nx0 = int(cols.min())
    nx1 = int(cols.max()) + 1
    new_width = nx1 - nx0

    if new_width <= rect.w:
        return rect
    if new_width >= int(width * HORIZONTAL_EXPAND_MAX_WIDTH_RATIO):
        return rect

    return BorderRect(nx0, rect.y, new_width, rect.h)


def _soft_threshold_bbox(gray: np.ndarray) -> BorderRect | None:
    """Build a larger fallback bbox using a slightly softer threshold."""

    blurred = cv2.GaussianBlur(gray, MASK_BLUR_KERNEL, 0)
    otsu_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    soft_threshold = max(0, int(otsu_value) - SOFT_BBOX_THRESHOLD_DELTA)
    _, binary = cv2.threshold(blurred, soft_threshold, 255, cv2.THRESH_BINARY)
    return _bbox_from_nonzero(binary)


def _needs_soft_bbox_fallback(rect: BorderRect, edge_bbox: BorderRect | None) -> bool:
    """Check if current rectangle is likely too small vs edge-based extent."""

    if edge_bbox is None:
        return False
    width_gap = edge_bbox.w - rect.w
    height_gap = edge_bbox.h - rect.h
    return (
        width_gap > int(rect.w * SOFT_FALLBACK_WIDTH_GAP_RATIO)
        or height_gap > int(rect.h * SOFT_FALLBACK_HEIGHT_GAP_RATIO)
    )


def _expand_rect_with_candidate(rect: BorderRect, candidate: BorderRect, image_shape: tuple[int, int]) -> BorderRect:
    """Merge current rectangle with a fallback candidate when changes are meaningful."""

    image_height, image_width = image_shape
    min_dx = max(CANDIDATE_MIN_DELTA_PIXELS, int(image_width * CANDIDATE_MIN_DELTA_RATIO))
    min_dy = max(CANDIDATE_MIN_DELTA_PIXELS, int(image_height * CANDIDATE_MIN_DELTA_RATIO))

    left = rect.x
    top = rect.y
    right = rect.x + rect.w
    bottom = rect.y + rect.h

    candidate_left = candidate.x
    candidate_top = candidate.y
    candidate_right = candidate.x + candidate.w
    candidate_bottom = candidate.y + candidate.h

    if rect.x - candidate_left >= min_dx:
        left = candidate_left
    if rect.y - candidate_top >= min_dy:
        top = candidate_top
    if candidate_right - right >= min_dx:
        right = candidate_right
    if candidate_bottom - bottom >= min_dy:
        bottom = candidate_bottom

    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return rect

    if width * height >= int(image_width * image_height * CANDIDATE_MAX_AREA_RATIO):
        return rect

    return BorderRect(left, top, width, height)


def _extend_bottom_with_soft_mask(rect: BorderRect, gray: np.ndarray, base_mask: np.ndarray) -> BorderRect:
    """Try to extend bottom boundary for edge cases with clipped lower border.

    This function is intentionally conservative:
    - It only runs when the current bottom already has enough evidence.
    - It extends downward only (keeps top/left/right stable).
    - It caps extra growth to avoid jumping to unrelated regions.
    """

    height, width = gray.shape
    bottom = rect.y + rect.h - 1
    if bottom < 0 or bottom >= height:
        return rect

    # Trigger only when base mask has substantial support at current bottom.
    base_row_counts = (base_mask > 0).sum(axis=1)
    trigger_threshold = max(BOTTOM_TRIGGER_MIN_PIXELS, int(rect.w * BOTTOM_TRIGGER_WIDTH_RATIO))
    if int(base_row_counts[bottom]) < trigger_threshold:
        return rect

    # Build a softer mask to pick up weak lower-edge pixels.
    blurred = cv2.GaussianBlur(gray, MASK_BLUR_KERNEL, 0)
    otsu_value, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    soft_threshold = max(0, int(otsu_value) - BOTTOM_SOFT_THRESHOLD_DELTA)
    _, soft_binary = cv2.threshold(blurred, soft_threshold, 255, cv2.THRESH_BINARY)

    pad_x = max(BOTTOM_PAD_X_MIN, int(rect.w * BOTTOM_PAD_X_RATIO))
    x0 = max(0, rect.x - pad_x)
    x1 = min(width, rect.x + rect.w + pad_x)
    roi = soft_binary[:, x0:x1]
    if roi.size == 0:
        return rect

    # Row occupancy over the horizontal span around current rectangle.
    row_counts = (roi > 0).sum(axis=1)
    min_pixels = max(BOTTOM_ROW_MIN_PIXELS, int((x1 - x0) * BOTTOM_ROW_MIN_RATIO))
    valid = row_counts >= min_pixels

    anchor = bottom
    if not valid[anchor]:
        search_start = max(0, bottom - BOTTOM_ANCHOR_SEARCH_RADIUS)
        search_end = min(height, bottom + BOTTOM_ANCHOR_SEARCH_RADIUS)
        nearby = np.where(valid[search_start:search_end])[0]
        if nearby.size == 0:
            return rect
        anchor = search_start + int(nearby[-1])

    end = anchor
    while end + 1 < height and valid[end + 1]:
        end += 1

    # If immediate continuity failed, allow a nearby sparse run bridge.
    if end <= bottom + BOTTOM_MIN_EXTENSION_PIXELS:
        sparse_pixels = max(BOTTOM_SPARSE_MIN_PIXELS, int((x1 - x0) * BOTTOM_SPARSE_RATIO))
        sparse_valid = row_counts >= sparse_pixels
        run_start: int | None = None
        best_end = end
        max_gap = BOTTOM_MAX_GAP_PIXELS
        min_run_len = BOTTOM_MIN_RUN_LENGTH
        peak_threshold = max(BOTTOM_PEAK_MIN_PIXELS, int(rect.w * BOTTOM_PEAK_WIDTH_RATIO))

        for idx in range(bottom + 1, height):
            if sparse_valid[idx]:
                if run_start is None:
                    run_start = idx
            elif run_start is not None:
                run_end = idx - 1
                run_len = run_end - run_start + 1
                run_peak = int(row_counts[run_start : run_end + 1].max())
                if (
                    run_start - bottom <= max_gap
                    and run_len >= min_run_len
                    and run_peak >= peak_threshold
                ):
                    best_end = max(best_end, run_end)
                run_start = None

        if run_start is not None:
            run_end = height - 1
            run_len = run_end - run_start + 1
            run_peak = int(row_counts[run_start : run_end + 1].max())
            if (
                run_start - bottom <= max_gap
                and run_len >= min_run_len
                and run_peak >= peak_threshold
            ):
                best_end = max(best_end, run_end)

        end = best_end

    if end <= bottom + BOTTOM_MIN_EXTENSION_PIXELS:
        return rect

    max_extra = max(BOTTOM_MAX_EXTRA_MIN_PIXELS, int(rect.h * BOTTOM_MAX_EXTRA_HEIGHT_RATIO))
    capped_end = min(end, bottom + max_extra)
    new_h = capped_end - rect.y + 1
    if new_h <= rect.h:
        return rect

    return BorderRect(rect.x, rect.y, rect.w, new_h)


def _extend_bottom_with_edges(rect: BorderRect, edges: np.ndarray) -> BorderRect:
    """Extend bottom boundary using edge support inside rectangle span."""

    height, width = edges.shape
    bottom = rect.y + rect.h - 1
    if bottom < 0 or bottom >= height:
        return rect

    pad_x = max(BOTTOM_PAD_X_MIN, int(rect.w * BOTTOM_PAD_X_RATIO))
    x0 = max(0, rect.x - pad_x)
    x1 = min(width, rect.x + rect.w + pad_x)
    roi = edges[:, x0:x1]
    if roi.size == 0:
        return rect

    row_counts = (roi > 0).sum(axis=1)
    min_pixels = max(BOTTOM_ROW_MIN_PIXELS, int((x1 - x0) * BOTTOM_ROW_MIN_RATIO))
    valid_rows = np.where(row_counts >= min_pixels)[0]
    if valid_rows.size == 0:
        return rect

    candidate_bottom = int(valid_rows.max())
    if candidate_bottom <= bottom + BOTTOM_MIN_EXTENSION_PIXELS:
        return rect

    max_extra = max(BOTTOM_MAX_EXTRA_MIN_PIXELS, int(rect.h * BOTTOM_MAX_EXTRA_HEIGHT_RATIO))
    capped_bottom = min(candidate_bottom, bottom + max_extra)
    if capped_bottom <= bottom:
        return rect

    return BorderRect(rect.x, rect.y, rect.w, capped_bottom - rect.y + 1)


def _extend_top_with_soft_bbox(rect: BorderRect, gray: np.ndarray) -> BorderRect:
    """Extend top boundary upward using soft-threshold evidence."""

    soft_bbox = _soft_threshold_bbox(gray)
    if soft_bbox is None:
        return rect

    top_gap = rect.y - soft_bbox.y
    if top_gap < TOP_MIN_EXTENSION_PIXELS:
        return rect

    max_extra = max(TOP_MAX_EXTRA_MIN_PIXELS, int(rect.h * TOP_MAX_EXTRA_HEIGHT_RATIO))
    extra = min(top_gap, max_extra)
    if extra <= 0:
        return rect

    new_top = max(0, rect.y - extra)
    new_height = (rect.y + rect.h) - new_top
    if new_height <= rect.h:
        return rect

    return BorderRect(rect.x, new_top, rect.w, new_height)


def detect_border_rect(
    image: np.ndarray,
    debug_dir: Path | None = None,
    image_name: str = "image",
) -> BorderRect | None:
    """Detect border rectangle for one image.

    Pipeline overview:
    1) Build edge map + binary masks.
    2) Extract/score contours and choose best candidate.
    3) Expand with edge/mask fallbacks for better outer boundary coverage.
    4) Optionally write debug artifacts.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = _prepare_edges(gray)
    raw_edge_bbox = _bbox_from_nonzero(edges)
    edge_bbox = _bbox_from_edge_density(edges)
    if edge_bbox is None:
        edge_bbox = raw_edge_bbox
    masks = _prepare_masks(gray)
    base_mask = masks[0]

    all_contours: list[np.ndarray] = []

    # Keep the mask with most foreground pixels for debug visualization.
    selected_mask = np.zeros_like(gray)
    for mask in masks:
        mask_contours, cleaned_mask = _contours_from_binary(mask)
        all_contours.extend(mask_contours)
        if np.count_nonzero(cleaned_mask) > np.count_nonzero(selected_mask):
            selected_mask = cleaned_mask

    contour = _best_contour(all_contours, gray.shape)
    if contour is None:
        # Fallback to edge bbox if no contour passes filters.
        edge_bbox = _bbox_from_edge_density(edges)
        if edge_bbox is None:
            edge_bbox = _bbox_from_nonzero(edges)
        if edge_bbox is None:
            return None
        rect = edge_bbox
    else:
        x, y, w, h = cv2.boundingRect(contour)
        rect = BorderRect(x, y, w, h)

    robust_edge_bbox = _bbox_from_edge_density(edges)
    if (
        robust_edge_bbox is not None
        and _is_suspiciously_wide(rect, gray.shape)
        and robust_edge_bbox.w
        <= rect.w
        - max(EDGE_BBOX_MIN_PIXELS, int(gray.shape[1] * WIDE_RECT_RESET_MIN_SHRINK_RATIO))
    ):
        rect = robust_edge_bbox

    rect = _expand_rect_with_edges(rect, edges)
    rect = _expand_rect_horizontally_with_mask(rect, base_mask)

    if _needs_soft_bbox_fallback(rect, edge_bbox):
        soft_bbox = _soft_threshold_bbox(gray)
        if soft_bbox is not None:
            candidate_edge_bbox = raw_edge_bbox if raw_edge_bbox is not None else edge_bbox
            if (
                candidate_edge_bbox is not None
                and not _touches_horizontal_edges(candidate_edge_bbox, gray.shape[1])
            ):
                soft_bbox = _expand_rect_with_candidate(soft_bbox, candidate_edge_bbox, gray.shape)
            rect = _expand_rect_with_candidate(rect, soft_bbox, gray.shape)

    rect = _extend_top_with_soft_bbox(rect, gray)
    rect = _extend_bottom_with_soft_mask(rect, gray, base_mask)
    rect = _extend_bottom_with_edges(rect, edges)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        contour_mask = np.zeros_like(gray)
        if contour is not None:
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        mask_for_debug = contour_mask if np.count_nonzero(contour_mask) > 0 else selected_mask
        _save_debug_images(image, edges, mask_for_debug, debug_dir, image_name)

        debug_image = image.copy()
        cv2.rectangle(
            debug_image,
            (rect.x, rect.y),
            (rect.x + rect.w - 1, rect.y + rect.h - 1),
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(debug_dir / f"{image_name}_final.jpg"), debug_image)

    return rect


def process_image(
    image_path: Path,
    output_dir: Path,
) -> BorderRect:
    """Process one image and write both text result and marked image.

    Outputs:
    - `<stem>_border.txt` containing `x y w h`
    - Marked image preserving original format (`.jpg` or `.bmp`)
    """

    image = _load_image(image_path)
    rect = detect_border_rect(image, debug_dir=_debug_dir(output_dir), image_name=image_path.stem)
    if rect is None:
        rect = BorderRect(0, 0, 0, 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{image_path.stem}_border.txt"
    output_path = output_dir / output_name
    output_path.write_text(rect.as_line(), encoding="utf-8")
    marked_image = image.copy()
    if rect.w > 0 and rect.h > 0:
        cv2.rectangle(
            marked_image,
            (rect.x, rect.y),
            (rect.x + rect.w - 1, rect.y + rect.h - 1),
            (0, 255, 0),
            2,
        )
    suffix = image_path.suffix if image_path.suffix else ".jpg"
    if suffix.lower() == ".bmp":
        marked_name = f"{image_path.stem}_mark_boder{suffix}"
    else:
        marked_name = f"{image_path.stem}_mark_border{suffix}"
    marked_path = output_dir / marked_name
    cv2.imwrite(str(marked_path), marked_image)
    return rect


def detect_image_border(image_path: Path) -> BorderRect:
    """Detect border rectangle for one image path without writing outputs."""

    image = _load_image(image_path)
    rect = detect_border_rect(image)
    if rect is None:
        return BorderRect(0, 0, 0, 0)
    return rect
