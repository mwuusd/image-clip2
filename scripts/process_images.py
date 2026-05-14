from __future__ import annotations

"""CLI entry point for batch/single-image phone border detection."""

import argparse
from pathlib import Path

from src.detect_border import detect_image_border
from src.detect_border import process_image
from src.slice_phone_region import slice_phone_region

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "phone_image"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "phone_border"
DEFAULT_CLIP_OUTPUT_DIR = PROJECT_ROOT / "phone_image_clip"


def _collect_images(input_dir: Path) -> list[Path]:
    """Collect supported image files from `input_dir`.

    Search is recursive so images in nested subdirectories are included.
    The pattern list intentionally includes upper/lower-case variants so behavior
    is consistent across case-sensitive and case-insensitive filesystems.
    """

    # Keep extension list explicit so supported formats are obvious.
    patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG", "*.bmp", "*.BMP")
    images: list[Path] = []
    for pattern in patterns:
        images.extend(input_dir.rglob(pattern))
    return sorted(set(images))


def _image_paths_for_output(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    clip_output_dir: Path,
) -> tuple[Path, Path, str]:
    """Return per-image output dirs and a display path.

    When `image_path` is inside `input_dir`, preserve its relative parent path in
    both output roots. Otherwise, write directly to output roots.
    """

    try:
        relative_path = image_path.resolve().relative_to(input_dir.resolve())
        relative_parent = relative_path.parent
        display_path = relative_path.as_posix()
    except ValueError:
        relative_parent = Path()
        display_path = image_path.name

    return output_dir / relative_parent, clip_output_dir / relative_parent, display_path


def _normalize_overlap(overlap: float) -> float:
    """Normalize overlap input as ratio in [0, 1).

    Accepts either ratio form (e.g., `0.1`) or percent form (e.g., `10`).
    """

    ratio = overlap / 100.0 if overlap >= 1 else overlap
    if ratio < 0 or ratio >= 1:
        msg = f"slice overlap must be in [0, 1) or [0, 100), got {overlap}"
        raise ValueError(msg)
    return ratio


def run() -> int:
    """Parse CLI arguments, process images, and return process exit code.

    Exit codes:
    - 0: success
    - 1: invalid input path, no files found, or at least one processing failure
    """

    parser = argparse.ArgumentParser(description="Detect phone borders in images.")
    parser.add_argument(
        "input_dir_positional",
        nargs="?",
        default=None,
        help="Optional input directory (same as --input-dir).",
    )
    parser.add_argument(
        "output_dir_positional",
        nargs="?",
        default=None,
        help="Optional output directory (same as --output-dir).",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory containing input images. Default: project phone_image.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write border .txt files. Default: project phone_border.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional single image path to process.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-b",
        "--border-only",
        action="store_true",
        help="Run boundary detection only.",
    )
    mode_group.add_argument(
        "-c",
        "--clip",
        action="store_true",
        help="Run boundary detection and clipping.",
    )
    parser.add_argument(
        "--clip-output-dir",
        default=None,
        help="Directory to write sliced phone-region images. Default: project phone_image_clip.",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=512,
        help="Base clip size in pixels for both dimensions. Default: 512.",
    )
    parser.add_argument(
        "--slice-width",
        type=int,
        default=None,
        help="Optional clip width in pixels. Overrides --slice-size for width.",
    )
    parser.add_argument(
        "--slice-height",
        type=int,
        default=None,
        help="Optional clip height in pixels. Overrides --slice-size for height.",
    )
    parser.add_argument(
        "--slice-overlap",
        type=float,
        default=20.0,
        help="Clip overlap as ratio (0-1) or percent (0-100). Default: 20.",
    )
    args = parser.parse_args()

    if args.border_only:
        mode = "border"
    elif args.clip:
        mode = "clip"
    else:
        mode = "slice"

    # `input_dir`/`output_dir` are normalized Path objects used everywhere below.
    if args.input_dir_positional is not None and args.input_dir is not None:
        print("Provide input directory via either positional path or --input-dir, not both.")
        return 1
    if mode in {"border", "clip"} and args.output_dir_positional is not None and args.output_dir is not None:
        print("Provide output directory via either positional path or --output-dir, not both.")
        return 1

    raw_input_dir = args.input_dir_positional or args.input_dir or str(DEFAULT_INPUT_DIR)
    raw_output_dir = args.output_dir_positional or args.output_dir or str(DEFAULT_OUTPUT_DIR)
    raw_clip_output_dir = args.clip_output_dir or str(DEFAULT_CLIP_OUTPUT_DIR)
    input_dir = Path(raw_input_dir).expanduser()
    output_dir = Path(raw_output_dir).expanduser()
    clip_output_dir = Path(raw_clip_output_dir).expanduser()

    do_slice = mode in {"clip", "slice"}

    slice_width = args.slice_size
    slice_height = args.slice_size
    slice_overlap = 0.0
    if do_slice:
        if args.slice_size <= 0:
            print(f"slice size must be > 0, got {args.slice_size}")
            return 1

        slice_width = args.slice_width if args.slice_width is not None else args.slice_size
        slice_height = args.slice_height if args.slice_height is not None else args.slice_size
        if slice_width <= 0:
            print(f"slice width must be > 0, got {slice_width}")
            return 1
        if slice_height <= 0:
            print(f"slice height must be > 0, got {slice_height}")
            return 1

        try:
            slice_overlap = _normalize_overlap(args.slice_overlap)
        except ValueError as exc:
            print(exc)
            return 1

    if args.image:
        image_path = Path(args.image).expanduser()
        image_output_dir, image_clip_output_dir, display_path = _image_paths_for_output(
            image_path,
            input_dir,
            output_dir,
            clip_output_dir,
        )
        try:
            if mode == "border":
                rect = process_image(image_path, image_output_dir)
            elif mode == "clip":
                rect = process_image(image_path, image_output_dir)
                slice_phone_region(
                    image_path=image_path,
                    rect=rect,
                    clip_output_dir=image_clip_output_dir,
                    slice_width=slice_width,
                    slice_height=slice_height,
                    overlap_ratio=slice_overlap,
                )
            else:
                rect = detect_image_border(image_path)
                slice_phone_region(
                    image_path=image_path,
                    rect=rect,
                    clip_output_dir=image_clip_output_dir,
                    slice_width=slice_width,
                    slice_height=slice_height,
                    overlap_ratio=slice_overlap,
                )
        except FileNotFoundError as exc:
            print(exc)
            return 1
        print(f"{display_path}: {rect.as_line()}")
        return 0

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}")
        return 1

    images = _collect_images(input_dir)
    if not images:
        print(f"No supported image files found in {input_dir}")
        return 1

    failures = 0
    for image_path in images:
        image_output_dir, image_clip_output_dir, display_path = _image_paths_for_output(
            image_path,
            input_dir,
            output_dir,
            clip_output_dir,
        )
        try:
            if mode == "border":
                rect = process_image(image_path, image_output_dir)
            elif mode == "clip":
                rect = process_image(image_path, image_output_dir)
                slice_phone_region(
                    image_path=image_path,
                    rect=rect,
                    clip_output_dir=image_clip_output_dir,
                    slice_width=slice_width,
                    slice_height=slice_height,
                    overlap_ratio=slice_overlap,
                )
            else:
                rect = detect_image_border(image_path)
                slice_phone_region(
                    image_path=image_path,
                    rect=rect,
                    clip_output_dir=image_clip_output_dir,
                    slice_width=slice_width,
                    slice_height=slice_height,
                    overlap_ratio=slice_overlap,
                )
        except FileNotFoundError as exc:
            # Continue batch processing even if one file fails.
            failures += 1
            print(f"{display_path}: ERROR {exc}")
            continue
        print(f"{display_path}: {rect.as_line()}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
