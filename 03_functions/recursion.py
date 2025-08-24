"""
recursion.py
Classic recursion: factorial, fibonacci, tree depth.
Includes PyTorch/TensorFlow recursive parameter counting.
"""

from functools import lru_cache
from typing import Any, Dict, Set
import torch, torch.nn as nn
import tensorflow as tf

print("\n=== Recursion: classic ===")

def factorial(n:int)->int:
    assert n>=0
    if n<=1: return 1
    return n*factorial(n-1)

print("factorial(5):", factorial(5))

@lru_cache(None)
def fib(n:int)->int:
    if n<=1: return n
    return fib(n-1)+fib(n-2)

print("fib(0..10):", [fib(i) for i in range(11)])

def tree_depth(d:Dict[str,Any])->int:
    if not isinstance(d,dict) or not d: return 0
    return 1+max(tree_depth(v) for v in d.values())

sample={"a":{"b":{"c":1}}, "x":{"y":2}}
print("tree_depth:", tree_depth(sample))

print("\n=== Recursion with PyTorch / TensorFlow ===")

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,8,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(8,16,3,padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier=nn.Linear(16,4)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.features(x)
        x=x.flatten(1)
        return self.classifier(x)

def count_params(module:nn.Module)->int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

torch_model=TinyCNN()
print("torch params:", count_params(torch_model))

def tf_count_params(layer:tf.keras.layers.Layer, visited:Set[int]|None=None)->int:
    if visited is None: visited=set()
    lid=id(layer)
    if lid in visited: return 0
    visited.add(lid)
    total=sum(int(tf.size(v)) for v in layer.trainable_variables)
    for sub in layer.submodules:
        if id(sub)!=lid:
            total+=tf_count_params(sub,visited)
    return total

tf_model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(8,3,padding="same",input_shape=(32,32,3)),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(16,3,padding="same"),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4)
])
print("tf params:", tf_count_params(tf_model))
