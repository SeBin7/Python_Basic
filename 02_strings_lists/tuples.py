"""
tuples.py
Deep dive into Python tuples + PyTorch/TensorFlow usages and project-style patterns.
"""

from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import torch
import tensorflow as tf


# =====================================
# 1) Python tuples: basics & patterns
# =====================================
print("\n=== Python tuples: basics ===")
t: Tuple[int, int, int] = (1, 2, 3)
print("t:", t)
print("t[0], t[-1]:", t[0], t[-1])
print("t[1:]:", t[1:])  # slicing -> tuple

print("\n=== Immutability ===")
# Attempt to modify a tuple element -> TypeError
try:
    t[0] = 999
except TypeError as e:
    print("cannot assign to item:", type(e).__name__)

print("\n=== Packing / Unpacking ===")
p = 10, 20, 30           # packing (parentheses optional)
a, b, c = p              # unpacking
print("p, a, b, c:", p, a, b, c)

print("\n=== Star-unpacking (variable tail) ===")
t2 = (1, 2, 3, 4, 5)
head, *mid, tail = t2    # mid is collected as a list
print("head, mid, tail:", head, mid, tail)

print("\n=== Swap (Pythonic) ===")
x, y = 3, 5
x, y = y, x
print("x, y:", x, y)

print("\n=== One-element tuple (comma required) ===")
one = (42)         # int
one_tuple = (42,)  # tuple
print("types ->", type(one).__name__, type(one_tuple).__name__)

print("\n=== Methods: count / index ===")
t3 = ("a", "b", "a", "c")
print("count('a'):", t3.count("a"))
print("index('b'):", t3.index("b"))

print("\n=== Sorting / generator vs tuple ===")
t4 = (3, 1, 2)
print("sorted(t4) -> list:", sorted(t4))
gen = (x*x for x in range(4))  # generator, not a tuple
print("gen ->", type(gen).__name__, list(gen))
t_from_gen = tuple(x*x for x in range(4))
print("tuple(...) ->", t_from_gen)

print("\n=== Nested tuple unpack ===")
pt = (10, (20, 30))
(xv, (yv, zv)) = pt
print("x, y, z:", xv, yv, zv)

print("\n=== Tuple vs List (immutable/mutable) ===")
t5 = (1, 2, 3)       # immutable
lst = [1, 2, 3]      # mutable
lst[0] = 9
print("t5, lst:", t5, lst)


# =====================================
# 2) PyTorch: tuple-heavy APIs & usage
# =====================================
print("\n=== PyTorch: tuple returns (split/LSTM), dataset (x,y) ===")

# torch.split -> returns a tuple of tensors
x = torch.arange(12).view(3, 4)  # (3,4)
a, b, c = torch.split(x, 1, dim=0)  # three (1,4) tensors
print("split shapes:", a.shape, b.shape, c.shape)

# LSTM: output, (h_n, c_n) tuple
seq_len, batch, in_dim, hid = 7, 2, 5, 4
lstm = torch.nn.LSTM(input_size=in_dim, hidden_size=hid, num_layers=1, batch_first=False)
inp = torch.randn(seq_len, batch, in_dim)
out, (h, c) = lstm(inp)
print("LSTM out:", out.shape, "| h:", h.shape, "| c:", c.shape)  # (S,B,H), (1,B,H), (1,B,H)

# Dataset: each sample is a (x, y) tuple -> DataLoader collates into (batch_x, batch_y)
class ToyTupleDataset(torch.utils.data.Dataset):
    def __len__(self) -> int: return 5
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.randn(4), torch.tensor(idx % 3)

loader = torch.utils.data.DataLoader(ToyTupleDataset(), batch_size=2, shuffle=False)
bx, by = next(iter(loader))
print("DataLoader tuple -> X shape:", bx.shape, "| y:", by.tolist())

# Model forward returns a tuple (logits, features)
class MLPReturnTuple(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 3)
        self.act = torch.nn.ReLU()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.act(self.fc1(x))
        logits = self.fc2(feat)
        return logits, feat  # (B,3), (B,8)

model = MLPReturnTuple()
logits, feat = model(bx)
loss = torch.nn.CrossEntropyLoss()(logits, by)
print("forward -> logits/feat:", logits.shape, feat.shape, "| CE loss:", float(loss.item()))


# =====================================
# 3) TensorFlow: tuples with tf.data & multi-output models
# =====================================
print("\n=== TensorFlow: (x,y) tuples in tf.data & multi-outputs ===")

# tf.data: stream of (features, labels) tuples
xs = tf.random.normal((6, 4))
ys = tf.constant([0, 1, 2, 0, 1, 2], dtype=tf.int32)
ds = tf.data.Dataset.from_tensor_slices((xs, ys)).batch(3)
batch_x, batch_y = next(iter(ds))
print("tf.data -> X shape:", batch_x.shape, "| y:", batch_y.numpy())

# Keras multi-output: model returns a tuple (logits, features)
inputs = tf.keras.Input(shape=(4,))
h = tf.keras.layers.Dense(8, activation="relu")(inputs)
logits = tf.keras.layers.Dense(3)(h)
model_tf = tf.keras.Model(inputs, (logits, h))  # outputs as a tuple
out_logits, out_feat = model_tf(batch_x, training=False)
print("Keras outputs -> logits/feat:", out_logits.shape, out_feat.shape)

# LSTM with return_state=True: (seq_out, h, c)
inp_tf = tf.random.normal((2, 7, 5))  # (B,T,F)
lstm_tf = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
seq_out, h_tf, c_tf = lstm_tf(inp_tf, training=False)
print("TF LSTM -> seq_out:", seq_out.shape, "| h:", h_tf.shape, "| c:", c_tf.shape)


# =====================================
# 4) Project-style: tuple keys (coords/transitions/sample-id) & dedup
# =====================================
print("\n=== Project-style: tuple keys (coords, transitions) ===")

# Use (x, y) tuple as a key for counting
points: List[Tuple[int, int]] = [(0,0), (1,1), (0,0), (2,1)]
coord_counts = Counter(points)
print("coord_counts.most_common:", coord_counts.most_common(2))

# Transition counting: (prev_label, curr_label) -> count
labels = ["dry", "dry", "wet", "snow", "wet", "wet"]
transition_counts: Dict[Tuple[str, str], int] = defaultdict(int)
for a, b in zip(labels, labels[1:]):
    transition_counts[(a, b)] += 1
print("transition_counts:", dict(transition_counts))

# Deduplicate with (video_id, frame_idx) tuple keys
seen: set[Tuple[str, int]] = set()
samples = [("vidA", 0), ("vidA", 1), ("vidA", 0), ("vidB", 3)]
unique_samples = []
for key in samples:
    if key not in seen:
        seen.add(key)
        unique_samples.append(key)
print("unique (video_id, frame):", unique_samples)

# Sort list of tuple pairs (lexicographic)
pairs = [(2, 0.9), (0, 0.8), (1, 0.95), (1, 0.6)]
print("sorted pairs:", sorted(pairs))  # class_id asc, then score asc

# Label ↔ id mapping as a list of (label, id) tuples
label2id = {"dry": 0, "wet": 1, "snow": 2}
pairs_li: List[Tuple[str, int]] = sorted(label2id.items(), key=lambda kv: kv[1])
print("label-id pairs:", pairs_li)

print("\n✅ Done.")
