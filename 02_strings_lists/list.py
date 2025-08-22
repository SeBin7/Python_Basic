"""
lists.py
Deep dive into Python lists, and converting lists to PyTorch / TensorFlow tensors.
"""

from typing import List
import copy
import torch
import tensorflow as tf

# =====================================
# 1) Python Lists (core, pitfalls, patterns)
# =====================================
print("\n=== Python Lists: Core ===")
fruits = ["apple", "banana", "cherry"]
print("Original:", fruits)

# Indexing & slicing
print("fruits[0]:", fruits[0])          # 'apple'
print("fruits[-1]:", fruits[-1])        # 'cherry'
print("fruits[1:3]:", fruits[1:3])      # ['banana','cherry']
print("fruits[::-1]:", fruits[::-1])    # reversed copy

# Modify (mutable)
fruits[1] = "blueberry"
print("After modify:", fruits)

# Append / extend / insert
fruits.append("date")
fruits.extend(["elderberry", "fig"])    # extend takes an iterable
fruits.insert(1, "pear")
print("After add:", fruits)

# Remove by value / pop by index
fruits.remove("apple")                   # first match
last = fruits.pop()                      # remove last
print("After remove + pop:", fruits, "| popped:", last)

# Membership / iteration / enumerate
print("'pear' in fruits:", "pear" in fruits)
for idx, f in enumerate(fruits):
    print(f"[enumerate] {idx} -> {f}")

# List comprehension (with condition)
squares = [x*x for x in range(8)]
evens   = [x for x in range(10) if x % 2 == 0]
print("squares:", squares)
print("evens:", evens)

# Nested comprehension (flatten a 2D list)
print("\n=== Python Lists: Nested & Flatten ===")
grid = [[1, 2, 3], [4, 5], [6]]
flat = [item for row in grid for item in row]
print("grid:", grid)
print("flat:", flat)

# Sort: in-place vs. sorted copy, key, reverse
print("\n=== Python Lists: Sorting ===")
nums = [5, 2, 9, 2, 1]
nums_sorted = sorted(nums)                 # new list
nums_desc   = sorted(nums, reverse=True)
words = ["car", "apple", "banana", "fig"]
words_by_len = sorted(words, key=len)
print("nums:", nums)
print("sorted(nums):", nums_sorted)
print("sorted(nums, reverse=True):", nums_desc)
print("sorted(words, key=len):", words_by_len)
nums.sort()                                # in-place
print("nums.sort() ->", nums)

# Shallow vs Deep copy
print("\n=== Python Lists: Copy Semantics ===")
a = [[1, 2], [3, 4]]
b_shallow = a[:]             # or list(a), copy.copy(a)
b_deep    = copy.deepcopy(a)
a[0][0] = 999
print("a:", a)
print("shallow copy affected:", b_shallow)   # inner list re-used
print("deep copy independent:", b_deep)

# Identity vs Equality
print("\n=== Python Lists: Identity vs Equality ===")
x = [1, 2, 3]
y = x
z = [1, 2, 3]
print("x == y:", x == y)     # True (values equal)
print("x is y:", x is y)     # True (same object)
print("x == z:", x == z)     # True
print("x is z:", x is z)     # False (different object)


# =====================================
# 2) PyTorch: list -> tensor (dtype, shape, pad)
# =====================================
print("\n=== PyTorch: list -> tensor ===")

# Basic conversion (homogeneous numeric lists)
data = [1, 2, 3, 4, 5]
t = torch.tensor(data)                 # dtype inferred (int64)
print("[Torch] tensor:", t, "| dtype:", t.dtype)

t_float = torch.tensor(data, dtype=torch.float32)
print("[Torch] float tensor:", t_float, "| dtype:", t_float.dtype)

print("[Torch] + 1:", t + 1)
print("[Torch] **2:", t ** 2)

# Nested list -> matrix/3D tensor (must be rectangular)
matrix = torch.tensor([[1, 2], [3, 4]])            # 2x2
cube   = torch.tensor([[[1, 2], [3, 4]],
                       [[5, 6], [7, 8]]])          # 2x2x2
print("[Torch] matrix:\n", matrix)
print("[Torch] cube shape:", cube.shape)

# Ragged lists: use padding (pad_sequence) after encoding to tensors
print("\n[Torch] Ragged -> pad_sequence")
seqs = [torch.tensor([1, 2, 3]),
        torch.tensor([4, 5]),
        torch.tensor([6])]
padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
print("padded:\n", padded)     # shape: (B, T_max)

# Stack vs cat (shape alignment rules)
print("\n[Torch] stack vs cat")
v1 = torch.tensor([1, 2, 3])
v2 = torch.tensor([4, 5, 6])
stacked = torch.stack([v1, v2], dim=0)    # add new dim -> (2, 3)
catted  = torch.cat([v1, v2], dim=0)      # concat along existing dim -> (6,)
print("stacked:\n", stacked, "| shape:", stacked.shape)
print("catted:\n", catted,  "| shape:", catted.shape)

# From Python lists to embeddings (common NLP pattern)
print("\n[Torch] list of ids -> Embedding")
ids = torch.tensor([[1, 2, 0], [3, 0, 0]])    # padded ids (B,T)
emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=4)
emb_out = emb(ids)
print("emb_out.shape:", emb_out.shape)        # (B,T,4)


# =====================================
# 3) TensorFlow: list -> tensor (dtype, ragged, pad)
# =====================================
print("\n=== TensorFlow: list -> tensor ===")

# Basic conversion
tf_data = [1, 2, 3, 4, 5]
tf_t = tf.constant(tf_data)                    # dtype inferred (int32)
tf_t_f = tf.constant(tf_data, dtype=tf.float32)
print("[TF] tensor:", tf_t, "| dtype:", tf_t.dtype)
print("[TF] float tensor:", tf_t_f, "| dtype:", tf_t_f.dtype)
print("[TF] + 1:", (tf_t + 1).numpy())
print("[TF] **2:", (tf_t ** 2).numpy())

# Nested list -> matrix / 3D tensor (rectangular)
tf_matrix = tf.constant([[1, 2], [3, 4]])
tf_cube   = tf.constant([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]])
print("[TF] matrix:\n", tf_matrix)
print("[TF] cube shape:", tf_cube.shape)

# Ragged tensors (variable-length sequences)
print("\n[TF] RaggedTensor + to_tensor (pad)")
rt = tf.ragged.constant([[1, 2, 3], [4, 5], [6]])  # ragged
print("ragged:", rt)
padded_tf = rt.to_tensor(default_value=0)          # pad to dense
print("padded:\n", padded_tf.numpy())

# Stack vs concat
print("\n[TF] stack vs concat")
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
stack_tf = tf.stack([a, b], axis=0)    # (2,3)
cat_tf   = tf.concat([a, b], axis=0)   # (6,)
print("stack:", stack_tf.numpy(), "| shape:", stack_tf.shape)
print("concat:", cat_tf.numpy(), "| shape:", cat_tf.shape)
