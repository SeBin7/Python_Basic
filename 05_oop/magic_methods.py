"""
magic_methods.py
Most useful Python magic methods with practical patterns:
- __repr__/__str__, __len__/__bool__, ordering (__eq__/__lt__), hashing
- arithmetic (__add__/__iadd__/__mul__/__rmul__), containers (__iter__/__getitem__/__contains__)
- __enter__/__exit__ (context manager), __call__
- PyTorch: Dataset(__len__/__getitem__), model __repr__/__len__, @ (matmul)
- TensorFlow: Keras Sequence(__len__/__getitem__), tensor @ (matmul)
"""

# =====================================
# 1) Python: repr/str, truthiness, ordering, hashing
# =====================================
from functools import total_ordering
from typing import Iterator

class Card:
    def __init__(self, rank: str, suit: str):
        self.rank, self.suit = rank, suit

    def __repr__(self) -> str:
        # unambiguous, for developers/logging
        return f"Card(rank={self.rank!r}, suit={self.suit!r})"

    def __str__(self) -> str:
        # pretty, for users
        return f"{self.rank} of {self.suit}"

c = Card("A", "Spades")
print(repr(c))  # Card(rank='A', suit='Spades')
print(str(c))   # A of Spades
print(c)        # A of Spades

class Bag:
    # small multiset-like container
    def __init__(self):
        self._data = {}

    def __len__(self) -> int:
        return sum(self._data.values())

    def __bool__(self) -> bool:
        # truthiness; empty bag is False
        return len(self) > 0

    def __contains__(self, item) -> bool:
        return item in self._data

    def __iter__(self) -> Iterator:
        # iterate over unique items
        return iter(self._data.keys())

    def __getitem__(self, item) -> int:
        return self._data.get(item, 0)

    def __setitem__(self, item, count: int) -> None:
        if count <= 0:
            self._data.pop(item, None)
        else:
            self._data[item] = int(count)

bag = Bag()
print(bool(bag))     # False
bag["apple"] = 2
bag["banana"] = 1
print(len(bag))      # 3
print("apple" in bag)  # True
print(bag["apple"])  # 2
print(list(bag))     # ['apple', 'banana']  (order may vary)

@total_ordering
class Vector2D:
    # value object with arithmetic + ordering by magnitude then angle
    __slots__ = ("x", "y")
    def __init__(self, x: float, y: float): self.x, self.y = float(x), float(y)
    def __repr__(self): return f"Vector2D({self.x:.2f}, {self.y:.2f})"
    def __str__(self):  return f"({self.x:.2f}, {self.y:.2f})"

    # arithmetic
    def __add__(self, other):  # v + u
        ox, oy = other.x, other.y
        return Vector2D(self.x + ox, self.y + oy)
    def __radd__(self, other):  #  u + v when u implements __add__? fallback
        return self.__add__(other)
    def __iadd__(self, other):  # v += u
        self.x += other.x; self.y += other.y; return self
    def __mul__(self, k: float):  # v * k
        return Vector2D(self.x * k, self.y * k)
    def __rmul__(self, k: float): # k * v
        return self.__mul__(k)

    # equality + ordering (for sort)
    def __eq__(self, other): return (self.x, self.y) == (other.x, other.y)
    def __lt__(self, other):
        # sort primarily by magnitude, secondarily by (x,y)
        self_mag = self.x*self.x + self.y*self.y
        other_mag = other.x*other.x + other.y*other.y
        return (self_mag, self.x, self.y) < (other_mag, other.x, other.y)

    # hashing (enable use as dict key / set element)
    def __hash__(self): return hash((round(self.x, 12), round(self.y, 12)))

v1, v2 = Vector2D(1, 2), Vector2D(3, -1)
print(v1)                 # (1.00, 2.00)
print(v1 + v2)            # (4.00, 1.00)
print(3 * v1)             # (3.00, 6.00)
v1 += Vector2D(1, 1)
print(v1)                 # (2.00, 3.00)
print(sorted([Vector2D(0,2), Vector2D(1,1), Vector2D(0,1)]))
# [Vector2D(0.00, 1.00), Vector2D(1.00, 1.00), Vector2D(0.00, 2.00)]

# context manager (with ...): __enter__/__exit__
import time

class Timer:
    def __init__(self, label="timer"): self.label = label
    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.label}] start")  # [block] start
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        print(f"[{self.label}] end: {dt:.3f}s")  # [block] end: ~0.xxxs
        return False  # propagate exceptions

with Timer("block"):
    _ = sum(range(100000))
# [block] start
# [block] end: 0.xxxs

# callable object: __call__
class Greeter:
    def __init__(self, prefix="Hi"): self.prefix = prefix
    def __call__(self, name: str) -> str:
        return f"{self.prefix}, {name}!"

g = Greeter("Hello")
print(g("Alice"))  # Hello, Alice!


# =====================================
# 2) PyTorch: Dataset(__len__/__getitem__), model __repr__/__len__, @ (matmul)
# =====================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, n=5, d=4, num_classes=3):
        self.X = torch.randn(n, d)
        self.y = torch.randint(0, num_classes, (n,))
    def __len__(self): return self.X.size(0)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

ds = ToyDataset(n=5, d=4, num_classes=3)
print(len(ds))            # 5
x0, y0 = ds[0]
print(x0.shape, int(y0))  # torch.Size([4]) <class int-like>

class ReportModule(nn.Module):
    # wraps an inner model; custom __repr__/__len__
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
    def forward(self, x): return self.inner(x)
    def __repr__(self):
        n_params = sum(p.numel() for p in self.parameters())
        return f"ReportModule(params={n_params}, inner={self.inner.__class__.__name__})"
    def __len__(self):
        # number of parameter tensors (or could return total params)
        return sum(1 for _ in self.parameters())

model = ReportModule(nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2)))
print(model)        # ReportModule(params=..., inner=Sequential)
print(len(model))   # e.g., 4  (weight/bias per Linear layer)

# matmul uses __matmul__ under the hood
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B
print(C.shape)      # torch.Size([2, 4])


# =====================================
# 3) TensorFlow: Sequence(__len__/__getitem__), tensor @ (matmul)
# =====================================
import tensorflow as tf
tf.random.set_seed(0)

class ToySequence(tf.keras.utils.Sequence):
    # minimal sequence to show __len__/__getitem__ for Keras
    def __init__(self, n=10, batch=4, d=4, num_classes=3):
        self.X = tf.random.normal((n, d))
        self.y = tf.random.uniform((n,), maxval=num_classes, dtype=tf.int32)
        self.batch = batch
    def __len__(self):  # number of batches
        return int(tf.math.ceil(tf.shape(self.X)[0] / self.batch))
    def __getitem__(self, idx):
        s = idx * self.batch
        e = tf.minimum((idx+1)*self.batch, tf.shape(self.X)[0])
        return self.X[s:e], self.y[s:e]

seq = ToySequence(n=10, batch=4, d=4, num_classes=3)
print(len(seq))      # 3
xb, yb = seq[0]
print(xb.shape, yb.shape)  # (4, 4) (4,)

# TensorFlow tensors also support @ (matmul)
A_tf = tf.random.normal((2, 3))
B_tf = tf.random.normal((3, 4))
C_tf = A_tf @ B_tf
print(C_tf.shape)    # (2, 4)
