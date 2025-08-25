"""
control_flow.py
Examples of control flow in Python, PyTorch, TensorFlow
+ Project-style snippets (PyTorch)
"""

import torch
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset

# =====================================
# 1) Python Control Flow
# =====================================

# if / elif / else
x = 10
if x > 10:
    print("[Python] x > 10")                 # (no output)
elif x == 10:
    print("[Python] x == 10")               # [Python] x == 10
else:
    print("[Python] x < 10")                 # (no output)

# for loop with range
for i in range(3):
    print("[Python] for i:", i)               # 0 -> 1 -> 2 (three lines)

# while loop with break / continue
count = 0
while count < 5:
    count += 1
    if count == 2:
        continue  # skip 2
    if count == 4:
        break     # stop at 4
    print("[Python] while count:", count)     # 1 -> 3 (two lines)

# for-else: else runs only if no break
for n in [1, 2, 3]:
    if n == 99:
        break
else:
    print("[Python] for-else finished")       # [Python] for-else finished

# while-else: else runs only if loop not broken
w = 0
while w < 2:
    w += 1
else:
    print("[Python] while-else finished")     # [Python] while-else finished

# nested loops + continue
for i in range(2):
    for j in range(2):
        if i == j:
            continue
        print("[Python] nested i,j:", i, j)    # (0,1) then (1,0)

# pass statement (placeholder)
val = -1
if val > 0:
    pass  # placeholder
else:
    print("[Python] pass example else")       # [Python] pass example else

# ternary expression (conditional expression)
y = 5
result = "positive" if y > 0 else "non-positive"
print("[Python] ternary result:", result)     # positive

# match-case (Python 3.10+)
status = 404
match status:
    case 200:
        print("[Python] OK")                   # (no output)
    case 404:
        print("[Python] Not Found")            # [Python] Not Found
    case _:
        print("[Python] Other")                # (no output)


# =====================================
# 2) PyTorch Control Flow with Tensors
# =====================================

# if with scalars vs. tensors
a = torch.tensor([1, 2, 3, 4, 5])
first = a[0]  # tensor scalar
if first.item() == 1:  # .item() to get Python scalar
    print("[Torch] first element is 1")       # [Torch] first element is 1

# iterate tensor
for v in a:
    print("[Torch] iter value:", v.item())     # 1 -> 2 -> 3 -> 4 -> 5

# boolean mask (comparison + logical ops)
mask = (a >= 3) & (a <= 4)
filtered = a[mask]
print("[Torch] mask:", mask)                   # tensor([False, False,  True,  True, False])
print("[Torch] filtered:", filtered)           # tensor([3, 4])

# simple dataloader loop
ds = TensorDataset(torch.arange(10).float(), torch.zeros(10).long())
dl = DataLoader(ds, batch_size=4, shuffle=False)
for step, (xb, yb) in enumerate(dl):
    print(f"[Torch] step={step}, xb={xb.tolist()}, yb={yb.tolist()}")  # steps 0..2


# =====================================
# 3) TensorFlow Control Flow with Tensors
# =====================================

# boolean mask
t = tf.constant([1, 2, 3, 4, 5])
tf_mask = (t >= 3) & (t <= 4)
tf_filtered = tf.boolean_mask(t, tf_mask)
print("[TF] mask:", tf_mask.numpy())           # [False False  True  True False]
print("[TF] filtered:", tf_filtered.numpy())   # [3 4]

# iterate tensor (eager mode)
for v in t:
    print("[TF] iter value:", int(v.numpy()))   # 1 -> 2 -> 3 -> 4 -> 5

# tf.data.Dataset loop
tf_ds = tf.data.Dataset.from_tensor_slices((tf.range(10, dtype=tf.float32),
                                            tf.zeros(10, dtype=tf.int32))).batch(4)
for step, (xb, yb) in enumerate(tf_ds):
    print(f"[TF] step={step}, xb={xb.numpy().tolist()}, yb={yb.numpy().tolist()}")  # steps 0..2


# =====================================
# 4) Project-style snippets (PyTorch)
#    – short, realistic patterns from Road Vision-style training
# =====================================

# 4-1) Thresholding + logical mask on feature maps (e.g., attention/activation regions)
feature_map = torch.randn(1, 1, 4, 4)  # (B, C, H, W)
hi = feature_map > 0.5
lo = feature_map < -0.5
roi = hi | lo  # keep high or low activations
print("[Proj] roi sum:", roi.sum().item())     # number of True positions (varies)

# 4-2) Compute accuracy with element-wise comparison
logits = torch.tensor([[0.2, 0.8, 0.0],
                       [0.6, 0.1, 0.3],
                       [0.1, 0.2, 0.7]])  # (B, C)
labels = torch.tensor([1, 0, 2])          # (B,)
preds = torch.argmax(logits, dim=1)
acc = (preds == labels).float().mean().item()
print("[Proj] acc:", round(acc, 3))            # 1.0

# 4-3) Minimal training loop skeleton with early stopping-like condition
model = torch.nn.Linear(8, 3)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

xb = torch.randn(32, 8)
yb = torch.randint(0, 3, (32,))

best_val = float("inf")
no_improve = 0
patience = 3

for epoch in range(20):
    # train step
    model.train()
    opt.zero_grad(set_to_none=True)
    out = model(xb)
    loss = loss_fn(out, yb)
    loss.backward()
    opt.step()

    # "validation" – here we reuse xb/yb as a placeholder
    model.eval()
    with torch.no_grad():
        val_out = model(xb)
        val_loss = loss_fn(val_out, yb).item()

    print(f"[Proj][epoch {epoch}] train={loss.item():.4f} val={val_loss:.4f}")  # values vary

    # early stop-like control flow
    if val_loss < best_val - 1e-4:
        best_val = val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print("[Proj] early stop triggered")  # may trigger depending on vals
            break

# 4-4) Occlusion-style region overwrite (slice + assignment)
img = torch.randn(1, 3, 8, 8)
y, x, h, w = 2, 3, 2, 2
img[:, :, y:y+h, x:x+w] = 0.0  # zero out a patch
print("[Proj] occluded patch mean:", img[:, :, y:y+h, x:x+w].mean().item())  # ~0.0 (exact value 0.0)
