# KernelVisualizer/main.py
# Headless-friendly CLI for applying CNN-style kernels to an image
# - Auto backend (Agg) for WSL2/no-GUI
# - Auto output path if --save omitted: <input_stem>_<kernel>.png
# - Visualization normalization: --viz none|abs|minmax
# Comments: English only

from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")  # headless backend for WSL2/servers

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# Relative imports inside the package
from .kernel_visualizer.kernels import get_kernel, list_kernels
from .kernel_visualizer.apply import conv2d
from .kernel_visualizer.visualize import show_side_by_side


def load_image(path: str, gray: bool = False) -> np.ndarray:
    """Load an image as np.ndarray (H×W or H×W×3/4)."""
    img = Image.open(path).convert("L" if gray else "RGBA")
    arr = np.array(img)
    # If RGBA but fully opaque, drop alpha
    if not gray and arr.ndim == 3 and arr.shape[2] == 4:
        if np.all(arr[..., 3] == 255):
            arr = arr[..., :3]
    return arr


def save_image(path: str, arr: np.ndarray) -> None:
    """Save np.ndarray as image."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def normalize_for_viz(arr: np.ndarray, mode: str) -> np.ndarray:
    """
    Normalize array for visualization-only:
      - 'none': no change
      - 'abs' : take absolute value then per-(channel) min-max to [0, 255]
      - 'minmax': per-(channel) min-max to [0, 255]
    Returns uint8 image.
    """
    if mode == "none":
        # Return uint8 if it's already uint8; else do a mild clamp/scale
        if arr.dtype == np.uint8:
            return arr
        x = np.clip(arr, 0.0, 1.0).astype(np.float32)
        return (x * 255.0).round().astype(np.uint8)

    x = arr.astype(np.float32)
    if mode == "abs":
        x = np.abs(x)

    if x.ndim == 2:
        mn, mx = x.min(), x.max()
        x = (x - mn) / (mx - mn + 1e-8)
    else:
        # per-channel min-max
        h, w, c = x.shape
        x2 = x.reshape(-1, c)
        mn = x2.min(axis=0)
        mx = x2.max(axis=0)
        x = (x - mn) / (mx - mn + 1e-8)

    return (np.clip(x, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Apply a CNN-style kernel to an image (headless-friendly).")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--kernel", required=True, choices=list_kernels(), help="Kernel name")
    parser.add_argument("--gray", action="store_true", help="Convert input to grayscale")
    parser.add_argument("--padding", default="same", choices=["same", "valid"])
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--border", default="reflect", choices=["reflect", "constant", "edge"])
    parser.add_argument("--save", default="", help="Output path. If empty => <input_stem>_<kernel>.png next to input.")
    parser.add_argument("--show", action="store_true", help="Try to show with matplotlib (GUI required)")
    parser.add_argument("--viz", default="none", choices=["none", "abs", "minmax"],
                        help="Visualization normalization for edge/derivative kernels")
    args = parser.parse_args()

    img = load_image(args.image, gray=args.gray)
    k = get_kernel(args.kernel)

    # Apply correlation-based conv (see apply.py).
    out = conv2d(
        img, k,
        padding=args.padding,
        stride=args.stride,
        border=args.border,
        keep_uint8=True,   # keep original dtype if it was uint8
        clip=True          # clip to [0,1] before uint8 scaling
    )

    # Visualization normalization (helps for Sobel/Laplacian, etc.)
    out_viz = normalize_for_viz(out, args.viz)

    # Decide save path
    if args.save:
        out_path = Path(args.save)
    else:
        inp = Path(args.image)
        out_path = inp.with_name(f"{inp.stem}_{args.kernel}.png")

    save_image(str(out_path), out_viz)

    # Show only if explicitly requested (GUI needed)
    if args.show:
        show_side_by_side(img, out_viz, title_left="Original", title_right=args.kernel)


if __name__ == "__main__":
    main()
