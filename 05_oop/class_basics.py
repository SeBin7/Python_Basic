"""
class_basics.py
Basic usage of Python classes: definition, __init__, self, attributes, methods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf

# =====================================
# 1) Python OOP Basics
# =====================================
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old."

# Create instances
p1 = Person("Alice", 25)
p2 = Person("Bob", 30)

print(p1.introduce())   # Hi, I'm Alice and I'm 25 years old.
print(p2.introduce())   # Hi, I'm Bob and I'm 30 years old.

# Add attribute dynamically
p1.job = "Engineer"
print(f"{p1.name}'s job: {p1.job}")  # Alice's job: Engineer


# Calculator class
class Calculator:
    def __init__(self, brand: str):
        self.brand = brand

    def add(self, x: int, y: int) -> int:
        return x + y

    def multiply(self, x: int, y: int) -> int:
        return x * y

calc = Calculator("Casio")
print("2 + 3 =", calc.add(2, 3))        # 2 + 3 = 5
print("4 * 5 =", calc.multiply(4, 5))  # 4 * 5 = 20


# Car class with __str__
class Car:
    def __init__(self, model: str, year: int):
        self.model = model
        self.year = year

    def __str__(self):
        return f"{self.year} {self.model}"

car1 = Car("Tesla Model 3", 2024)
print(car1)   # 2024 Tesla Model 3


# =====================================
# 2) PyTorch OOP Example (nn.Module)
# =====================================

class SimpleNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNN(4, 8, 2)
x = torch.randn(1, 4)    # input tensor
out = model(x)
print("PyTorch model output:", out)  # tensor([[... , ...]], grad_fn=<AddmmBackward0>)


# =====================================
# 3) TensorFlow OOP Example (tf.keras.Model)
# =====================================

class TFNN(tf.keras.Model):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation="relu")
        self.fc2 = tf.keras.layers.Dense(output_dim)

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)

tf_model = TFNN(8, 2)
x_tf = tf.random.normal((1, 4))
out_tf = tf_model(x_tf)
print("TensorFlow model output:", out_tf)  # tf.Tensor([... , ...], shape=(1, 2), dtype=float32)
