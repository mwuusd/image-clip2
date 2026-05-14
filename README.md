# image-clip

Detect phone borders in JPG images using OpenCV and output a bounding rectangle.

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
".venv\Scripts\python" -m pip install --upgrade pip
".venv\Scripts\pip" install -r requirements.txt
```

## Usage

### Process all images in a directory

By default (no `-b`/`-c`), process all supported image files (JPG, PNG, BMP)
in `phone_image` including nested subdirectories, and write slices to
`phone_image_clip`:

```bash
".venv\Scripts\python" -m scripts.process_images
```

You can also pass the input directory as a positional argument:

```bash
".venv\Scripts\python" -m scripts.process_images panel
".venv\Scripts\python" -m scripts.process_images -c panel
".venv\Scripts\python" -m scripts.process_images panel panel_border
```

This creates phone-region slices in `phone_image_clip` by default.
For nested inputs, output folders mirror the input subdirectory structure.

### Process a single image

Process a specific image file:

```bash
".venv\Scripts\python" -m scripts.process_images --image "phone_image/your_image.jpg"
```

### Select processing mode

- `-b` / `--border-only`: run boundary detection only
- `-c` / `--clip`: run boundary detection and tiling
- Default (no mode flag): run tiling mode without writing border outputs
- In default mode, `--output-dir` is ignored

```bash
".venv\Scripts\python" -m scripts.process_images -b
".venv\Scripts\python" -m scripts.process_images -c
".venv\Scripts\python" -m scripts.process_images
".venv\Scripts\python" -m scripts.process_images -b --image "phone_image/your_image.jpg"
".venv\Scripts\python" -m scripts.process_images -c --image "phone_image/your_image.jpg"
```

### Use custom directories

Process images from a different directory:

```bash
".venv\Scripts\python" -m scripts.process_images --input-dir "phone_image" --clip-output-dir "phone_image_clip"
".venv\Scripts\python" -m scripts.process_images -b --input-dir "phone_image" --output-dir "phone_border"
".venv\Scripts\python" -m scripts.process_images -c --input-dir "phone_image" --output-dir "phone_border" --clip-output-dir "phone_image_clip"
python -m scripts.process_images --input-dir "H:\crack_images_20260422\images_original" --clip-output-dir "H:\crack_images_20260422\images_slices" --slice-width 3072 --slice-height 3072 --slice-overlap 10
```

Use either positional input dir or `--input-dir` (not both).
You can also pass output dir positionally as the second argument.

In `-b`/`-c` modes, use either positional output dir or `--output-dir` (not both).
In default mode (no `-b`/`-c`), output-dir values are ignored.

Default directories (project root):

- `--input-dir`: `phone_image`
- `--output-dir`: `phone_border`
- `--clip-output-dir`: `phone_image_clip`

Example mirrored paths for nested input:

- Input: `phone_image/a/b/example.jpg`
- Border outputs: `phone_border/a/b/example_border.txt` and marked image in `phone_border/a/b/`
- Clip outputs: `phone_image_clip/a/b/example/example_c_1_x_100_y_200_w_512_h_512.jpg`, etc.

### Adjust slice behavior

These options are used in clip mode (`-c` or default mode).

By default, the detected phone region is sliced into `512x512` clips with `20%`
overlap and written to
`phone_image_clip/<relative_subdir>/<image_name>/` (or `phone_image_clip/<image_name>/`
for root-level images). Slices include context outside the detected boundary
using a per-side padding based on overlap.

```bash
".venv\Scripts\python" -m scripts.process_images --slice-size 512 --slice-overlap 20 --clip-output-dir "phone_image_clip"
```

- `--slice-size`: base clip size for both width and height (default: `512`)
- `--slice-width`: optional non-square clip width override
- `--slice-height`: optional non-square clip height override
- `--slice-overlap`: overlap ratio (`0.2`) or percent (`20`) (default: `20`)
  - Also controls outside-boundary context: per-side pad is
    `round(slice_width * overlap)` horizontally and
    `round(slice_height * overlap)` vertically.

Clip filename format:

- `<image_name>_c_<index>_x_<abs_x>_y_<abs_y>_w_<w>_h_<h>.<original_ext>`
- `x`/`y`: top-left slice coordinates in the original image
- `w`/`h`: actual slice width/height in pixels
- `c`: slice sequence number
- A slice index CSV is also written: `<image_name>_slice_index.csv`
- CSV columns: `file_name,x,y,w,h,c`

Example non-square slicing:

```bash
".venv\Scripts\python" -m scripts.process_images --slice-width 640 --slice-height 512 --slice-overlap 20
```

### Example workflow

1. Place your image files (JPG, PNG, BMP) in `phone_image`
2. Run the CLI:

```bash
".venv\Scripts\python" -m scripts.process_images
```

3. If you run `-b` or `-c`, check results in `phone_border` (mirrors input subdirectories):
   - `<image_name>_border.txt` - bounding rectangle coordinates
   - For JPG/PNG input: `<image_name>_mark_border.<original_ext>`
   - For BMP input: `<image_name>_mark_boder.bmp`
4. Check slices in `phone_image_clip/<relative_subdir>/<image_name>/`:
   - `<image_name>_c_1_x_<abs_x>_y_<abs_y>_w_<w>_h_<h>.<original_ext>`
   - `<image_name>_c_2_x_<abs_x>_y_<abs_y>_w_<w>_h_<h>.<original_ext>`
   - `<image_name>_slice_index.csv`
   - ...

### Running in PowerShell

If you're using PowerShell, run the following commands:

Do not copy prompt symbols like `PS ...>` or `$` into the command.

```powershell
& ".venv\Scripts\python" -m scripts.process_images
& ".venv\Scripts\python" -m scripts.process_images -c
& ".venv\Scripts\python" -m scripts.process_images -b
& ".venv\Scripts\python" -m scripts.process_images --image "phone_image/your_image.jpg"
& ".venv\Scripts\python" -m scripts.process_images --help
```

## Output format

Each output file is named `"<original name>_border.txt"` and contains one line:

```
x y w h
```

If no contour is detected, the output will be `0 0 0 0`.

## Visual output

For easier inspection, the CLI also writes:

- For JPG/PNG input, a marked image named
  `"<original name>_mark_border.<original_ext>"` with a green rectangle.
- For BMP input, a marked image named `"<original name>_mark_boder.bmp"`.

## Parameter Tuning Guide

The detector parameters live in `src/detect_border.py` as constants near the
top of the file. These are the most useful ones to tune:

- `HORIZONTAL_EXPAND_PIXEL_RATIO` and `HORIZONTAL_EXPAND_MIN_PIXELS`:
  Increase to make left/right expansion stricter; decrease to make it wider.
- `EDGE_EXPAND_PAD_RATIO` and `EDGE_EXPAND_MIN_PAD`:
  Increase to search farther around the initial rectangle before edge-based
  expansion.
- `SOFT_FALLBACK_WIDTH_GAP_RATIO` and `SOFT_FALLBACK_HEIGHT_GAP_RATIO`:
  Decrease to trigger soft-threshold fallback more often when bounds look small.
- `SOFT_BBOX_THRESHOLD_DELTA`:
  Increase to use a softer threshold and potentially recover larger boundaries.
- `BOTTOM_SOFT_THRESHOLD_DELTA` and `BOTTOM_MAX_EXTRA_HEIGHT_RATIO`:
  Useful when the bottom edge is clipped; increase carefully to avoid noise.
- `MIN_CONTOUR_AREA_RATIO` and `MAX_CONTOUR_AREA_RATIO`:
  Adjust contour size filtering if images have very small or very large phones.

Recommended tuning workflow:

1. Change one constant at a time.
2. Run one problematic image with `--image`.
3. Compare `phone_border/debug/*_final.jpg` against expected bounds.
4. Re-run full batch after single-image behavior is correct.

### Common Failure Patterns

- **Bottom boundary too small**
  - Increase `BOTTOM_SOFT_THRESHOLD_DELTA` first.
  - Then increase `BOTTOM_MAX_EXTRA_HEIGHT_RATIO` if still clipped.
  - If extension does not trigger, lower `BOTTOM_TRIGGER_MIN_PIXELS` or
    `BOTTOM_TRIGGER_WIDTH_RATIO` slightly.

- **Top boundary too small**
  - Increase `SOFT_BBOX_THRESHOLD_DELTA` a little.
  - Decrease `SOFT_FALLBACK_HEIGHT_GAP_RATIO` so fallback triggers earlier.
  - If still tight, increase `EDGE_EXPAND_PAD_RATIO`.

- **Left/right boundary too small**
  - Decrease `HORIZONTAL_EXPAND_PIXEL_RATIO` first.
  - Then decrease `HORIZONTAL_EXPAND_MIN_PIXELS` if needed.
  - If fallback is not activating, decrease `SOFT_FALLBACK_WIDTH_GAP_RATIO`.

- **Box expands too much into background**
  - Decrease `SOFT_BBOX_THRESHOLD_DELTA`.
  - Increase `HORIZONTAL_EXPAND_PIXEL_RATIO` and `HORIZONTAL_EXPAND_MIN_PIXELS`.
  - Decrease `BOTTOM_MAX_EXTRA_HEIGHT_RATIO`.

- **Detector picks wrong object/noisy region**
  - Increase `MIN_CONTOUR_AREA_RATIO` to ignore small objects.
  - Tighten aspect filtering by narrowing
    `ASPECT_GOOD_MIN`/`ASPECT_GOOD_MAX` and `ASPECT_OK_MIN`/`ASPECT_OK_MAX`.
  - Reduce edge growth by lowering `EDGE_EXPAND_PAD_RATIO`.
