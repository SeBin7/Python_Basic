"""
functions_basics.py
Core Python function concepts: def/return, *args/**kwargs, keyword-only args,
first-class functions, closures, decorator (timeit).
Includes small PyTorch / TensorFlow bridges showing first-class functions,
closures, and decorators in ML contexts.
"""

from __future__ import annotations
from typing import Any, Callable, List, Dict
from functools import wraps
import time

import torch
import torch.nn as nn
import tensorflow as tf

print("\n=== Functions: basics ===")

# --- Basic function definition / call / return ---
def add(a: int, b: int = 10) -> int:
    return a + b

print("add(3):", add(3))
print("add(3, 4):", add(3, 4))

# --- Variable arguments (*args, **kwargs) ---
def summarize(*nums: float, **opts: Any) -> float:
    return sum(nums) * opts.get("scale", 1.0)

print("summarize(1,2,3):", summarize(1,2,3))
print("summarize(1,2,3, scale=0.5):", summarize(1,2,3, scale=0.5))

# --- Keyword-only arguments ---
def clip(x: float, *, low: float, high: float) -> float:
    return max(low, min(high, x))

print("clip(3.5, low=0, high=1):", clip(3.5, low=0, high=1))

# --- First-class functions ---
def apply_all(x: Any, funcs: List[Callable[[Any], Any]]) -> List[Any]:
    return [f(x) for f in funcs]

sqr = lambda t: t*t
cube = lambda t: t*t*t
print("apply_all(3, [sqr, cube]):", apply_all(3, [sqr, cube]))

# --- Closures ---
def make_multiplier(k: float) -> Callable[[float], float]:
    def mul(x: float) -> float:
        return k * x
    return mul

times3 = make_multiplier(3)
print("times3(7):", times3(7))

# --- Decorator ---
def timeit(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            dt = (time.perf_counter()-t0)*1000
            print(f"[timeit] {fn.__name__} took {dt:.2f} ms")
    return wrapper

@timeit
def slow_sum(n: int) -> int:
    total=0
    for i in range(n):
        total += i
    return total

print("slow_sum(100000):", slow_sum(100_000))

# ============================================================
# PyTorch bridge: functions/closures/decorators in practice
# ============================================================
print("\n=== PyTorch: function registry, closure train_step ===")

# Function registry (first-class functions + lambda)
activations: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": lambda z: torch.clamp_min(z, 0.0),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
}
z = torch.linspace(-2, 2, steps=5)
print("activations['relu'](z):", activations["relu"](z))

# Simple model
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(8, 4),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = TinyCNN()

# Closure: make a one-step trainer bound to a given loss function
def make_train_step(loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    def train_step(xb: torch.Tensor, yb: torch.Tensor) -> float:
        model.train(); opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward(); opt.step()
        return float(loss.item())
    return train_step

criterion = nn.CrossEntropyLoss()
train_step = make_train_step(criterion)

xb = torch.randn(8, 3, 32, 32)
yb = torch.randint(0, 4, (8,))
print("torch train_step loss:", train_step(xb, yb))

# Decorator: time a forward call
@timeit
def forward_once(xb: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(xb)

_ = forward_once(torch.randn(4,3,32,32))

# ============================================================
# TensorFlow bridge: @tf.function and HOF
# ============================================================
print("\n=== TensorFlow: @tf.function, train_step factory ===")

# @tf.function acts like a decorator for graph compilation
@tf.function
def tf_forward(m: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    return m(x, training=False)

tf_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, 3, padding="same", input_shape=(32,32,3)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4)
])

# Higher-order function returning a compiled train step
def tf_make_train_step(loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]):
    opt = tf.keras.optimizers.SGD(0.1)
    @tf.function
    def train_step(xb: tf.Tensor, yb: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            logits = tf_model(xb, training=True)
            loss = loss_fn(y_true=yb, y_pred=logits)
        grads = tape.gradient(loss, tf_model.trainable_variables)
        opt.apply_gradients(zip(grads, tf_model.trainable_variables))
        return loss
    return train_step

tf_step = tf_make_train_step(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
tx = tf.random.normal((8,32,32,3))
ty = tf.random.uniform((8,), maxval=4, dtype=tf.int32)
print("tf train_step loss:", float(tf_step(tx, ty).numpy()))
