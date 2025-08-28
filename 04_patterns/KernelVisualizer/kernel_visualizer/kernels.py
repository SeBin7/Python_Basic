# kernels.py
# Canonical 2D convolution kernels for CV/CNN intuition

from __future__ import annotations
import numpy as np
from typing import Dict, Iterable

def _as_np(arr: Iterable[Iterable[float]]) -> np.ndarray:
    """Convert nested iterable to float32 numpy array."""
    return np.asarray(arr, dtype=np.float32)


# --- 3x3 canonical kernels ----------------------------------------------------
def identity_3x3() -> np.ndarray:
    return _as_np([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ])


def outline_3x3() -> np.ndarray:
    # Similar to Laplacian but keeps center positive
    return _as_np([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1],
    ])


def laplacian_3x3() -> np.ndarray:
    # Isotropic edge detector (second derivative)
    return _as_np([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0],
    ])


def sobel_x() -> np.ndarray:
    # Horizontal gradient (responds to vertical edges)
    return _as_np([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])


def sobel_y() -> np.ndarray:
    # Vertical gradient (responds to horizontal edges)
    return _as_np([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1],
    ])


def sharpen_3x3() -> np.ndarray:
    # Unsharp mask style sharpening
    return _as_np([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0],
    ])


def emboss_3x3() -> np.ndarray:
    # Emboss/relief effect
    return _as_np([
        [-2, -1, 0],
        [-1,  1, 1],
        [ 0,  1, 2],
    ])


def box_blur_3x3() -> np.ndarray:
    k = np.ones((3, 3), dtype=np.float32)
    return k / k.sum()  # 1/9


def gaussian_3x3() -> np.ndarray:
    # Discrete Gaussian ~ sigma≈1
    k = _as_np([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ])
    return k / k.sum()  # 1/16


# --- 5x5 options (a bit smoother) --------------------------------------------
def gaussian_5x5() -> np.ndarray:
    # Pascal weights outer product -> binomial Gaussian approx
    v = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    k = np.outer(v, v)  # 5x5
    return k / k.sum()  # 1/256


def box_blur_5x5() -> np.ndarray:
    k = np.ones((5, 5), dtype=np.float32)
    return k / k.sum()  # 1/25

# --- Registry & helpers -------------------------------------------------------
def _build_registry() -> Dict[str, np.ndarray]:
    """Materialize a dictionary of named kernels (immutable arrays)."""
    reg = {
        # 3x3
        "identity_3x3": identity_3x3(),
        "outline_3x3": outline_3x3(),
        "laplacian_3x3": laplacian_3x3(),
        "sobel_x": sobel_x(),
        "sobel_y": sobel_y(),
        "sharpen_3x3": sharpen_3x3(),
        "emboss_3x3": emboss_3x3(),
        "box_blur_3x3": box_blur_3x3(),
        "gaussian_3x3": gaussian_3x3(),
        # 5x5
        "gaussian_5x5": gaussian_5x5(),
        "box_blur_5x5": box_blur_5x5(),
    }
    # Ensure all are float32 and read-only
    for k, v in reg.items():
        reg[k] = np.asarray(v, dtype=np.float32)
        reg[k].setflags(write=False)
    return reg


_KERNELS: Dict[str, np.ndarray] = _build_registry()


def get_kernel(name: str) -> np.ndarray:
    """Return a copy of the kernel by name (defensive copy for safety)."""
    if name not in _KERNELS:
        raise KeyError(f"Unknown kernel name: {name}. "
                       f"Available: {', '.join(sorted(_KERNELS))}")
    return _KERNELS[name].copy()


def list_kernels() -> list[str]:
    """List available kernel names."""
    return sorted(_KERNELS.keys())