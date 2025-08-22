"""
strings.py
String handling in Python, then bridging to PyTorch and TensorFlow.
"""

from typing import List, Dict
import torch
import tensorflow as tf

# =====================================
# 1) Python Strings
# =====================================
print("\n=== Python Strings ===")

# Declaration
s = "Hello, World!"
print("s:", s)

# Indexing & slicing
print("s[0]:", s[0])           # 'H'
print("s[-1]:", s[-1])         # '!'
print("s[7:12]:", s[7:12])     # 'World'
print("s[::-1]:", s[::-1])     # reversed

# Common methods
print("lower():", s.lower())
print("upper():", s.upper())
print("replace():", s.replace("World", "Python"))
print("split(','):", s.split(","))
print("strip():", "  spaced  ".strip())

# Search
print("find('World'):", s.find("World"))
print("startswith('Hello'):", s.startswith("Hello"))
print("endswith('!'):", s.endswith("!"))

# Join
words = ["fast", "campus", "rocks"]
print("'-'.join(words):", "-".join(words))

# Formatting
name, score = "Sebin", 97.5
print(f"f-string: {name} scored {score:.1f}")
print("format(): {} scored {}".format(name, score))
print("percent: %s scored %.1f" % (name, score))

# Immutability demo
s2 = s + " :)"
print("concatenation:", s2)
print("original still same:", s)

# Multiline
multi = """Line1
Line2
Line3"""
print("multiline:\n", multi)


# =====================================
# 2) PyTorch: Working *with* strings (no native string tensor)
# =====================================
print("\n=== PyTorch: Strings via encoding/tokenization ===")

# NOTE:
# PyTorch does not have a native string dtype. Typical workflow:
# 1) Keep text as Python str
# 2) Encode or tokenize to integers (bytes/char/word ids)
# 3) Convert to torch.Tensor for models (embeddings/RNN/Transformer)

text = "hello"
batch_texts = ["hi", "hello", "he"]

# A) Byte encoding (UTF-8) → Tensor of ints
byte_ids = list(text.encode("utf-8"))          # e.g., [104, 101, 108, 108, 111]
byte_tensor = torch.tensor(byte_ids, dtype=torch.uint8)
print("byte_tensor:", byte_tensor)

# Decode back to string
decoded = bytes(byte_tensor.tolist()).decode("utf-8")
print("decoded:", decoded)

# B) Simple char-level indexing with safe vocab
# Build vocab from ALL texts and include special tokens
all_text = "".join(batch_texts + [text])
vocab = {"<pad>": 0, "<unk>": 1}
for ch in sorted(set(all_text)):
    if ch not in vocab:
        vocab[ch] = len(vocab)
itos = {i: t for t, i in vocab.items()}
print("vocab:", vocab)

# Safe encoder using <unk> fallback
def encode_chars(s: str) -> torch.Tensor:
    return torch.tensor([vocab.get(c, vocab["<unk>"]) for c in s], dtype=torch.long)

# Encode and pad to same length
encoded = [encode_chars(t) for t in batch_texts]
padded = torch.nn.utils.rnn.pad_sequence(
    encoded, batch_first=True, padding_value=vocab["<pad>"]
)
print("encoded (list of tensors):", encoded)
print("padded (batch,char_ids):\n", padded)

# Example: pass through an embedding
emb = torch.nn.Embedding(num_embeddings=len(vocab), embedding_dim=4)
emb_out = emb(padded)  # shape: (B, T, 4)
print("emb_out.shape:", emb_out.shape)



# =====================================
# 3) TensorFlow: tf.strings utilities
# =====================================
print("\n=== TensorFlow: tf.strings ===")

tf_s = tf.constant("Hello, TensorFlow!")
print("tf_s:", tf_s.numpy().decode())

# Lower/Upper
print("lower():", tf.strings.lower(tf_s).numpy().decode())
print("upper():", tf.strings.upper(tf_s).numpy().decode())

# Split (to RaggedTensor)
spl = tf.strings.split(tf_s, sep=" ")
print("split:", spl)  # RaggedTensor of words

# Regex replace
clean = tf.strings.regex_replace(tf_s, pattern=r"[^A-Za-z ]", rewrite="")
print("regex_clean:", clean.numpy().decode())

# Lengths
print("len bytes:", tf.strings.length(tf_s).numpy())
print("len unicode:", tf.strings.length(tf_s, unit="UTF8_CHAR").numpy())

# Batch strings + pad to dense
tf_batch = tf.constant(["hi", "hello", "he"])
tf_chars = tf.strings.unicode_split(tf_batch, "UTF-8")     # RaggedTensor [B, T]
print("unicode_split:", tf_chars)

# Map each character to code points (ints)
code_points = tf.strings.unicode_decode(tf_batch, "UTF-8") # RaggedTensor
print("unicode_decode:", code_points)

# Convert Ragged → Dense with padding (0)
dense_codes = code_points.to_tensor(default_value=0)
print("dense_codes:\n", dense_codes.numpy())

# Join back per-example
joined = tf.strings.reduce_join(tf_chars, axis=1, separator="")
print("joined:", joined.numpy().tolist())
