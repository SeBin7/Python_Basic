# visualize.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def show_attention(mat: np.ndarray, title="Attention", save: str | None = None, show: bool = False):
    plt.imshow(mat, cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save)
        print(f"[INFO] Saved attention pattern to {save}")

    if show:
        plt.show()  # Only if GUI available
    plt.close()
