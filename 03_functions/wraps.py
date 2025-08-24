"""
decorator_wraps_compare.py
Compare decorators with/without functools.wraps in:
1) Pure Python functions
2) PyTorch forward pass
3) TensorFlow forward pass
"""

import time
from functools import wraps
import torch
import torch.nn as nn
import tensorflow as tf


# ============================================================
# Decorator definitions
# ============================================================

# --- Decorator without @wraps ---
# -> Works, but __name__ / __doc__ of the original function are lost.
def timeit_no_wraps(fn):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        print(f"[no_wraps] {fn.__name__} took {(time.time()-t0)*1000:.2f} ms")
        return out
    return wrapper

# --- Decorator with @wraps ---
# -> Preserves metadata (function name, docstring, annotations).
def timeit_with_wraps(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = fn(*args, **kwargs)
        print(f"[with_wraps] {fn.__name__} took {(time.time()-t0)*1000:.2f} ms")
        return out
    return wrapper


# ============================================================
# 1) Pure Python demo
# ============================================================

@timeit_no_wraps
def slow_sum_no_wraps(n: int):
    """Sum from 0 to n-1 (no_wraps)."""
    return sum(range(n))

@timeit_with_wraps
def slow_sum_with_wraps(n: int):
    """Sum from 0 to n-1 (with_wraps)."""
    return sum(range(n))

print("\n--- Pure Python: check __name__ and __doc__ ---")
print("no_wraps name:", slow_sum_no_wraps.__name__)
print("no_wraps doc :", slow_sum_no_wraps.__doc__)
print("with_wraps name:", slow_sum_with_wraps.__name__)
print("with_wraps doc :", slow_sum_with_wraps.__doc__)

_ = slow_sum_no_wraps(100_000)
_ = slow_sum_with_wraps(100_000)


# ============================================================
# 2) PyTorch demo: measure decorator behavior on model forward
# ============================================================

class TinyMLP(nn.Module):
    """A tiny feedforward network for demonstration."""
    def __init__(self, d_in=64, d_hidden=128, d_out=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

torch_model = TinyMLP()

@timeit_no_wraps
def torch_forward_no_wraps(xb: torch.Tensor) -> torch.Tensor:
    """PyTorch forward pass (no_wraps)."""
    with torch.no_grad():
        return torch_model(xb)

@timeit_with_wraps
def torch_forward_with_wraps(xb: torch.Tensor) -> torch.Tensor:
    """PyTorch forward pass (with_wraps)."""
    with torch.no_grad():
        return torch_model(xb)

print("\n--- PyTorch: check __name__ ---")
print("no_wraps name:", torch_forward_no_wraps.__name__)
print("with_wraps name:", torch_forward_with_wraps.__name__)

xb = torch.randn(1024, 64)
_ = torch_forward_no_wraps(xb)
_ = torch_forward_with_wraps(xb)


# ============================================================
# 3) TensorFlow demo: measure decorator behavior on model forward
# ============================================================

tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(64,)),
    tf.keras.layers.Dense(4)
])

@timeit_no_wraps
def tf_forward_no_wraps(xb: tf.Tensor) -> tf.Tensor:
    """TensorFlow forward pass (no_wraps)."""
    return tf_model(xb, training=False)

@timeit_with_wraps
def tf_forward_with_wraps(xb: tf.Tensor) -> tf.Tensor:
    """TensorFlow forward pass (with_wraps)."""
    return tf_model(xb, training=False)

print("\n--- TensorFlow: check __name__ ---")
print("no_wraps name:", tf_forward_no_wraps.__name__)
print("with_wraps name:", tf_forward_with_wraps.__name__)

tx = tf.random.normal((1024, 64))
_ = tf_forward_no_wraps(tx)
_ = tf_forward_with_wraps(tx)
