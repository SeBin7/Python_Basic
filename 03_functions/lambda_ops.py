"""
lambda_ops.py
Lambda expressions + map/filter/sorted in three styles (def/lambda/list-comp),
plus common lambda idioms. Includes small Torch/TF snippets to show usage.
"""

from typing import List, Tuple, Callable
import torch
import tensorflow as tf

print("\n=== Lambda: map/filter/sorted ===")

nums: List[int] = [5,2,9,1,5,6]
pairs: List[Tuple[str,int]] = [("alice",3), ("bob",1), ("carol",2)]

# map
def double(x:int)->int: return x*2
print("map (def + map):", list(map(double, nums)))
print("map (lambda + map):", list(map(lambda x:x*2, nums)))
print("map (list comprehension):", [x*2 for x in nums])

# filter
def is_even(x:int)->bool: return x%2==0
print("filter (def + filter):", list(filter(is_even, nums)))
print("filter (lambda + filter):", list(filter(lambda x:x%2==0, nums)))
print("filter (list comprehension):", [x for x in nums if x%2==0])

# sorted
def by_second(item:Tuple[str,int])->int: return item[1]
print("sorted (def + sorted):", sorted(pairs, key=by_second))
print("sorted (lambda + sorted):", sorted(pairs, key=lambda x:x[1]))
print("sorted (list comprehension):", [p for p in sorted(pairs, key=lambda x:x[1])])

# Common lambda idioms
print("\n=== Handy lambda patterns ===")
print("square via lambda:", list(map(lambda x:x**2, nums)))
print("cube via lambda:", list(map(lambda x:x**3, nums)))
relu_like = lambda x: x if x>0 else 0
print("relu-like transform:", [relu_like(x) for x in [-2,0,3]])
print("sorted by abs:", sorted([3,-10,2,-1], key=lambda x: abs(x)))
print("multi-key sort:", sorted(pairs, key=lambda t:(t[1], t[0])))

# ----------------------------------------------------
# Tiny Torch/TF snippets using lambda
# ----------------------------------------------------
print("\n=== Torch/TF with lambda ===")

# Torch: activation registry via lambda
activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": lambda z: torch.clamp_min(z, 0.0),
    "leaky": lambda z: torch.where(z>0, z, 0.01*z),
}
z = torch.linspace(-2,2,5)
print("torch relu via lambda:", activations["relu"](z))

# TF: map_fn with lambda (character count per string)
batch = tf.constant(["hi","hello","he"])
lengths = tf.map_fn(lambda s: tf.strings.length(s), batch, fn_output_signature=tf.int32)
print("tf lengths via map_fn:", lengths.numpy().tolist())
