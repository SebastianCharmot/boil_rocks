"""Read an image `after.JPG`, remove green pixels, and show before/after.

Behavior:
- Loads an image file (default: after.JPG in the same folder).
- Detects green pixels using a simple rule: green channel significantly
  higher than red/blue and above a brightness threshold.
- Produces two outputs:
  - `after_no_green.png` (RGBA) with green pixels made transparent
  - `after_no_green.jpg` (RGB) where green pixels are replaced with white
- Displays the original and processed images side-by-side.

Usage:
	python process.py [path/to/after.JPG]

Requirements: pillow, numpy, matplotlib
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
from PIL import Image


def load_image(path: str) -> Image.Image:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Image not found: {path}")
	img = Image.open(path).convert("RGBA")
	return img


def remove_green_pixels(img: Image.Image, ratio: float = 1.2, diff_thresh: int = 40) -> Image.Image:
	"""Return a new RGBA image where detected green pixels are transparent.

	Detection rule (per-pixel):
	  G > ratio * R and G > ratio * B and (G - max(R, B)) > diff_thresh

	These parameters work well for strong green screens; adjust for
	different lighting or tint.
	"""
	arr = np.array(img)  # shape (H, W, 4)
	if arr.ndim != 3 or arr.shape[2] < 3:
		raise ValueError("Unsupported image shape")

	r = arr[..., 0].astype(np.int16)
	g = arr[..., 1].astype(np.int16)
	b = arr[..., 2].astype(np.int16)

	# Compute green mask
	mask = (g > (ratio * r)) & (g > (ratio * b)) & ((g - np.maximum(r, b)) > diff_thresh)

	# Create output RGBA array
	out = arr.copy()
	# Set alpha to 0 where mask is True
	out[..., 3] = np.where(mask, 0, out[..., 3])

	return Image.fromarray(out)


def save_results(orig: Image.Image, processed: Image.Image, out_base: str = "after_no_green") -> Tuple[str, str]:
	"""Save processed image as PNG (with transparency) and JPG (green->white).

	Returns tuple (png_path, jpg_path)
	"""
	png_path = f"{out_base}.png"
	jpg_path = f"{out_base}.jpg"

	# Save PNG with transparency
	processed.save(png_path)

	# For JPG, composite over white background
	white_bg = Image.new("RGB", processed.size, (255, 255, 255))
	rgb = Image.alpha_composite(white_bg.convert("RGBA"), processed).convert("RGB")
	rgb.save(jpg_path, quality=95)

	return png_path, jpg_path


def show_side_by_side(orig: Image.Image, processed: Image.Image, title: str = "Before / After") -> None:
	try:
		import matplotlib.pyplot as plt

		fig, axes = plt.subplots(1, 2, figsize=(12, 6))
		axes[0].imshow(orig.convert("RGB"))
		axes[0].set_title("Original")
		axes[0].axis("off")

		axes[1].imshow(processed)
		axes[1].set_title("No green (transparent)")
		axes[1].axis("off")

		fig.suptitle(title)
		plt.tight_layout()
		plt.show()
	except Exception as exc:  # pragma: no cover - interactive display may fail in CI
		print("Could not display images interactively:", exc)


def main(argv: list[str] | None = None) -> int:
	argv = argv if argv is not None else sys.argv[1:]
	img_path = argv[0] if argv else "after.JPG"

	print(f"Loading image: {img_path}")
	try:
		orig = load_image(img_path)
	except Exception as e:
		print("Error loading image:", e)
		return 2

	processed = remove_green_pixels(orig, ratio=1.2, diff_thresh=40)

	png_path, jpg_path = save_results(orig, processed)
	print(f"Saved: {png_path}")
	print(f"Saved: {jpg_path}")

	# Show images side-by-side
	show_side_by_side(orig, processed)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

