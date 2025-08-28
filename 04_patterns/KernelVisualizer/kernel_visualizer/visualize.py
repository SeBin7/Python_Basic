# visualize.py
# Simple matplotlib helpers to compare original vs filtered

from __future__ import annotations
import numpy as np               # <-- 이게 빠져 있었음
import matplotlib.pyplot as plt

def show_side_by_side(orig: np.ndarray, filt: np.ndarray, title_left="Original", title_right="Filtered"):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if orig.ndim == 2:
        ax1.imshow(orig, cmap="gray")
    else:
        ax1.imshow(orig)
    ax1.set_title(title_left)
    ax1.axis("off")

    if filt.ndim == 2:
        ax2.imshow(filt, cmap="gray")
    else:
        ax2.imshow(filt)
    ax2.set_title(title_right)
    ax2.axis("off")

    plt.tight_layout()
    #plt.show()
    plt.close(fig)   # <-- prevents block
