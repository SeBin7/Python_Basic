# apply.py
# Minimal 2D correlation (not flipped) with padding/stride, gray/RGB support
# Comments: English only

from __future__ import annotations
import numpy as np
from typing import Literal, Tuple

PadMode = Literal["same", "valid"]
BorderMode = Literal["reflect", "constant", "edge"]  # np.pad modes


def _to_float_img(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Cast to float32 in [0,1] if uint8, otherwise float32. Return also flag 'was_uint8'."""
    if img.dtype == np.uint8:
        return (img.astype(np.float32) / 255.0), True
    return img.astype(np.float32), False


def _pad2d(x: np.ndarray, pad: Tuple[int, int], mode: BorderMode = "reflect") -> np.ndarray:
    """Pad HxW with given pad=(ph, pw) on both sides."""
    ph, pw = pad
    if ph == 0 and pw == 0:
        return x
    return np.pad(x, ((ph, ph), (pw, pw)), mode=mode)


def _conv2d_single_channel(
    x: np.ndarray,
    k: np.ndarray,
    padding: PadMode = "same",
    stride: int = 1,
    border: BorderMode = "reflect",
) -> np.ndarray:
    """
    2D correlation on single-channel image.
    x: HxW float32, k: khxkw float32
    Returns: out_h x out_w (float32)
    Note: This is cross-correlation (kernel not flipped). Good enough for visualization.
    """
    assert x.ndim == 2, "x must be HxW"
    assert k.ndim == 2, "k must be khxkw"
    kh, kw = k.shape
    H, W = x.shape

    if padding == "same":
        # same output size for stride=1
        ph = kh // 2
        pw = kw // 2
        xp = _pad2d(x, (ph, pw), mode=border)
    elif padding == "valid":
        xp = x
        ph = pw = 0
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    Hp, Wp = xp.shape
    out_h = (Hp - kh) // stride + 1
    out_w = (Wp - kw) // stride + 1

    out = np.empty((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        ih = i * stride
        for j in range(out_w):
            jw = j * stride
            patch = xp[ih:ih + kh, jw:jw + kw]
            out[i, j] = float((patch * k).sum())
    return out


def conv2d(
    img: np.ndarray,
    kernel: np.ndarray,
    padding: PadMode = "same",
    stride: int = 1,
    border: BorderMode = "reflect",
    keep_uint8: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """
    Apply 2D correlation to grayscale or RGB image.

    img: HxW or HxWx3 (uint8/float)
    kernel: khxkw (float32)
    padding: 'same' | 'valid'
    stride: int >= 1
    border: np.pad mode for borders (reflect|constant|edge)
    keep_uint8: if input was uint8, return uint8 (with [0,255] scaling)
    clip: clip outputs to [0,1] (before uint8 scaling) for stability

    Returns:
        np.ndarray of shape HxW or HxWx3
    """
    x, was_u8 = _to_float_img(img)
    k = kernel.astype(np.float32)

    if x.ndim == 2:
        y = _conv2d_single_channel(x, k, padding=padding, stride=stride, border=border)
    elif x.ndim == 3 and x.shape[2] in (3, 4):
        # process first 3 channels; if 4 (RGBA), pass alpha through
        ch = x.shape[2]
        out_chans = []
        for c in range(min(3, ch)):
            out_chans.append(_conv2d_single_channel(x[..., c], k, padding, stride, border))
        y = np.stack(out_chans, axis=-1)
        if ch == 4:
            # alpha passthrough (resized if needed)
            if y.shape[:2] == x.shape[:2]:
                alpha = x[..., 3:4]
            else:
                # for stride>1/valid, crop alpha to match (naive center crop)
                oh, ow = y.shape[:2]
                alpha = x[:oh, :ow, 3:4]
            y = np.concatenate([y, alpha], axis=-1)
    else:
        raise ValueError("img must be HxW or HxWx(3|4)")

    if clip:
        y = np.clip(y, 0.0, 1.0)

    if keep_uint8 and was_u8:
        y = (y * 255.0).round().astype(np.uint8)
    return y
