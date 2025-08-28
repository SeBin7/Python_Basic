import numpy as np

def identity_attention(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.float32)

def uniform_attention(n: int) -> np.ndarray:
    return np.full((n, n), 1/n, dtype=np.float32)

def band_attention(n: int, k: int = 1) -> np.ndarray:
    mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(max(0, i-k), min(n, i+k+1)):
            mat[i, j] = 1.0
    # row-wise normalize
    mat /= mat.sum(axis=1, keepdims=True)
    return mat
