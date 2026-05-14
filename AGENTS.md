# Agent Guide for image-clip

This repository provides a small OpenCV-based CLI for detecting phone borders
in JPG images and writing bounding rectangle coordinates.

## Quick Start

- Python version: 3.9+ recommended
- Virtualenv location: `.venv`
- Main entry point: `scripts/process_images.py`

## Build / Lint / Test

This project does not currently use a build system, lint, or test runner. Use
the commands below for basic execution and validation.

- Create venv:
  - `python -m venv .venv`
  - `".venv\Scripts\python" -m pip install --upgrade pip`
  - `".venv\Scripts\pip" install -r requirements.txt`
- Run CLI (batch):
  - `".venv\Scripts\python" -m scripts.process_images`
- Run CLI (single image):
  - `".venv\Scripts\python" -m scripts.process_images --image "~/phone_image/example.jpg"`
- Run CLI with custom dirs:
  - `".venv\Scripts\python" -m scripts.process_images --input-dir "~/phone_image" --output-dir "~/phone_border"`

### Single Test Equivalent

There is no test suite. To validate one image, use the single-image command
above and verify the output `.txt` file in `~/phone_border`.

## Code Style Guidelines

Follow these conventions for any changes or new code.

### Imports

- Standard library first, third-party next, local imports last.
- One import per line, except for `from x import a, b` when concise.
- Avoid wildcard imports.

### Formatting

- Use 4 spaces for indentation.
- Keep lines <= 100 chars when practical.
- Use double quotes for strings unless avoiding escapes.
- Prefer f-strings for string interpolation.

### Types

- Use type hints for public functions and non-trivial helpers.
- Prefer `Path` from `pathlib` for file paths.
- Use `| None` for optional returns (Python 3.10+ style) when available.

### Naming

- Modules: `snake_case.py`.
- Functions: `snake_case` verbs.
- Classes: `CapWords`.
- Constants: `UPPER_SNAKE_CASE`.
- Private helpers: prefix with `_`.

### Error Handling

- Raise explicit exceptions for unrecoverable issues (e.g., missing image).
- For expected runtime issues (no contours), return a safe default and log.
- Do not swallow exceptions silently.

### File and Directory Behavior

- Input images are in `~/phone_image`.
- Output `.txt` files go to `~/phone_border`.
- Output format is a single line: `x y w h`.
- Create output directories if missing.
- Write a marked JPG `"<original name>_mark_border.jpg"` with a green rectangle.

### OpenCV Processing

- Use grayscale + blur + Canny + contours for border detection.
- Choose the largest contour assuming one phone per image.
- Compute bounding rectangle via `cv2.boundingRect`.

### CLI Behavior

- Use `argparse` and provide clear help text.
- Support a single-image mode via `--image`.
- Print per-image results to stdout for easy inspection.

## Repo Notes

- No `.cursor/rules`, `.cursorrules`, or `.github/copilot-instructions.md`
  are present at the repository root.
- If such rules are added later, incorporate them here.
