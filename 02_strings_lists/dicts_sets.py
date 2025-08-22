"""
dicts_sets.py
Deep dive into Python dicts & sets + PyTorch/TensorFlow usages and project-style patterns.
"""

from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import copy
import torch
import tensorflow as tf

# =====================================
# 1) Python dicts: CRUD, views, merge, sort, copy, comp
# =====================================
print("\n=== Python dicts: basics ===")
scores: Dict[str, int] = {"alice": 90, "bob": 85, "chris": 92}
print("scores:", scores)
print("scores['alice']:", scores["alice"])

# CRUD
scores["dave"] = 88          # create
scores["alice"] = 95         # update
print("after add/update:", scores)

# Safe access
print("get('zoe', default=-1):", scores.get("zoe", -1))

# setdefault
print("setdefault('emma', 80):", scores.setdefault("emma", 80))
print("after setdefault:", scores)

# pop/popitem
removed = scores.pop("bob")          # remove by key, return value
print("pop('bob') ->", removed, "|", scores)
k, v = scores.popitem()              # remove last (insertion-ordered)
print("popitem() ->", (k, v), "|", scores)

# update/merge
more = {"frank": 77, "gina": 83}
scores.update(more)
print("after update:", scores)

# dict merge operator (3.9+)
merged = scores | {"alice": 100, "harry": 70}
print("merged (|):", merged)

# Views
print("keys:", list(scores.keys()))
print("values:", list(scores.values()))
print("items:", list(scores.items()))

# Sorting by key / value
print("\n=== Python dicts: sorting ===")
by_key_asc  = dict(sorted(scores.items(), key=lambda kv: kv[0]))
by_val_desc = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
print("by_key_asc:", by_key_asc)
print("by_val_desc:", by_val_desc)

# Copy semantics
print("\n=== Python dicts: copy semantics ===")
conf = {"model": {"hidden": 128, "dropout": 0.1}, "lr": 1e-3}
shallow = conf.copy()
deep    = copy.deepcopy(conf)
conf["model"]["hidden"] = 256
print("conf:", conf)
print("shallow affected:", shallow)  # inner dict shared
print("deep independent:", deep)

# Dict comprehension
print("\n=== Python dicts: comprehension ===")
nums = [1, 2, 3, 4, 5]
squares_map = {x: x * x for x in nums if x % 2 == 1}
print("squares_map (odd only):", squares_map)


# =====================================
# 2) Python sets: ops, relations, frozenset, pitfalls
# =====================================
print("\n=== Python sets: ops & relations ===")
A = {1, 2, 3}
B = {3, 4, 5}
print("A|B (union):", A | B)
print("A&B (intersection):", A & B)
print("A-B (difference):", A - B)
print("A^B (symmetric diff):", A ^ B)

# subset/superset
print("A <= A|B:", A <= (A | B))
print("A < A|B:",  A <  (A | B))
print("A >= {1,2}:", A >= {1, 2})

# frozenset (hashable set)
fs = frozenset({1, 2, 3})
map_with_frozenset_key = {fs: "group123"}
print("frozenset as key:", map_with_frozenset_key)

# Pitfall: unhashable types as set/dict keys
print("hashable str ok as key:", {"abc": 1})
# list as key -> TypeError (unhashable), tuple ok if elements are hashable
# {"[1,2]": 1}  # strings are fine; but { [1,2]: 1 } would fail


# =====================================
# 3) Collections helpers: defaultdict, Counter
# =====================================
print("\n=== collections: defaultdict & Counter ===")
dd = defaultdict(list)
pairs = [("a", 1), ("b", 2), ("a", 3)]
for k, v in pairs:
    dd[k].append(v)
print("defaultdict(list):", dict(dd))

cnt = Counter(["dry", "wet", "dry", "snow", "dry"])
print("Counter:", cnt)
print("most_common:", cnt.most_common(2))


# =====================================
# 4) PyTorch: dict usage (state_dict, save/load, batch of dict)
# =====================================
print("\n=== PyTorch: state_dict & batching dict samples ===")

# state_dict save/load
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 3),
)
sd: Dict[str, torch.Tensor] = model.state_dict()
print("state_dict keys:", list(sd.keys())[:4], "...")

# save/load demo (in-memory)
buf = torch.save(sd, "tmp_state.pth")  # file write
loaded = torch.load("tmp_state.pth")
model.load_state_dict(loaded)
print("loaded ok:", True)

# batch of samples as list[dict] -> collate
samples = [
    {"id": "a", "x": torch.randn(4), "y": torch.tensor(1)},
    {"id": "b", "x": torch.randn(4), "y": torch.tensor(0)},
    {"id": "c", "x": torch.randn(4), "y": torch.tensor(2)},
]

def simple_collate(batch: List[Dict]):
    ids = [b["id"] for b in batch]
    xs  = torch.stack([b["x"] for b in batch], dim=0)  # (B,4)
    ys  = torch.stack([b["y"] for b in batch], dim=0)  # (B,)
    return {"id": ids, "x": xs, "y": ys}

batched = simple_collate(samples)
print("batched['id']:", batched["id"])
print("batched['x'].shape:", batched["x"].shape, "| batched['y'].shape:", batched["y"].shape)


# =====================================
# 5) TensorFlow: vocab lookup, feature dicts with tf.data
# =====================================
print("\n=== TensorFlow: lookup & tf.data with dict ===")

# StaticVocabularyTable with OOV bucket
vocab_list = ["<pad>", "dry", "wet", "snow"]
init = tf.lookup.KeyValueTensorInitializer(
    keys=vocab_list,
    values=tf.range(len(vocab_list), dtype=tf.int64),
)
table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets=1)
labels = tf.constant(["dry", "unknown", "snow"])
label_ids = table.lookup(labels)
print("label_ids:", label_ids.numpy())

# tf.data with dict elements
def gen():
    yield {"id": b"a", "x": tf.random.normal((4,)), "y": tf.constant(1, tf.int32)}
    yield {"id": b"b", "x": tf.random.normal((4,)), "y": tf.constant(0, tf.int32)}
ds = tf.data.Dataset.from_generator(gen, output_signature={
    "id": tf.TensorSpec(shape=(), dtype=tf.string),
    "x":  tf.TensorSpec(shape=(4,), dtype=tf.float32),
    "y":  tf.TensorSpec(shape=(), dtype=tf.int32),
}).batch(2)
for batch in ds.take(1):
    print("tf.data batch keys:", list(batch.keys()))
    print("batch['x'].shape:", batch["x"].shape, "| batch['y']:", batch["y"].numpy())


# =====================================
# 6) Project-style: label maps, class weights, confusion dict
# =====================================
print("\n=== Project-style: label maps, class weights, confusion ===")

# Label maps
label2id = {"dry": 0, "wet": 1, "snow": 2}
id2label = {i: l for l, i in label2id.items()}
print("label2id:", label2id, "| id2label:", id2label)

# Class weights from counts (to balance loss)
counts = Counter(["dry", "dry", "wet", "snow", "dry", "wet"])
num_classes = len(label2id)
weights = torch.zeros(num_classes, dtype=torch.float32)
total = sum(counts.values())
for lbl, cnt in counts.items():
    weights[label2id[lbl]] = total / (cnt * num_classes)
print("class weights:", weights.tolist())

# Example loss with weights (CrossEntropyLoss)
criterion = torch.nn.CrossEntropyLoss(weight=weights)
logits = torch.randn(5, num_classes)         # (B,C)
true_y = torch.tensor([0, 1, 0, 2, 0])       # (B,)
loss = criterion(logits, true_y)
print("weighted CE loss:", float(loss.item()))

# Confusion matrix via dict-of-dict
pred = torch.argmax(logits, dim=1).tolist()
true = true_y.tolist()
conf: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
for t, p in zip(true, pred):
    conf[id2label[t]][id2label[p]] += 1
print("confusion:")
for t_label, row in conf.items():
    print("  ", t_label, dict(row))
