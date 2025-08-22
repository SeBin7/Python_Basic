"""
variables.py
Examples of Python variables, PyTorch tensors, and TensorFlow tensors
"""

import torch
import tensorflow as tf

# =====================================
# 1. Python Variables
# =====================================
print("\n=== Python Variables ===")

# Variable declaration
integer_number = 10        # int
floating_number = 3.14     # float
message = "Hello"          # string
flag = True                # boolean

print("Values:", integer_number, floating_number, message, flag)

# Check variable types
print("Types:", type(integer_number), type(floating_number), type(message), type(flag))

# Examples
x, y, z = 10, "Hello", 3.14   # Multiple assignment
p = q = r = 100               # Same value assignment
m, n = 5, 10
m, n = n, m                   # Swapping

print("Assignments:", x, y, z)
print("Same value:", p, q, r)
print("Swapped:", m, n)

# Dynamic typing
var = 42
print("Dynamic typing (int):", var, type(var))
var = "text"
print("Dynamic typing (str):", var, type(var))

# Constants convention (by naming convention only)
PI = 3.14159
MAX_USERS = 100
print("Constants:", PI, MAX_USERS)


# =====================================
# 2. PyTorch Tensors
# =====================================
print("\n=== PyTorch Tensors ===")

# Basic tensors
scalar_t = torch.tensor(5)
vector_t = torch.tensor([1, 2, 3])
matrix_t = torch.tensor([[1, 2], [3, 4]])

print("Scalar:", scalar_t)
print("Vector:", vector_t)
print("Matrix:\n", matrix_t)

# Predefined tensors
print("Ones:\n", torch.ones(2, 3))
print("Zeros:\n", torch.zeros(2, 3))
print("Full:\n", torch.full((2, 3), 7))

# Random tensors
print("Randn (Normal):\n", torch.randn(2, 3))
print("Rand (Uniform):\n", torch.rand(2, 3))

# Tensor properties
randn_t = torch.randn(2, 3)
print("Properties -> Shape:", randn_t.shape, ", Dtype:", randn_t.dtype, ", Device:", randn_t.device)


# =====================================
# 3. TensorFlow Tensors
# =====================================
print("\n=== TensorFlow Tensors ===")

# Basic tensors
scalar_tf = tf.constant(5)
vector_tf = tf.constant([1, 2, 3])
matrix_tf = tf.constant([[1, 2], [3, 4]])

print("Scalar:", scalar_tf)
print("Vector:", vector_tf)
print("Matrix:\n", matrix_tf)

# Predefined tensors
print("Ones:\n", tf.ones((2, 3)))
print("Zeros:\n", tf.zeros((2, 3)))
print("Full:\n", tf.fill((2, 3), 7))

# Random tensors
print("Normal:\n", tf.random.normal((2, 3)))
print("Uniform:\n", tf.random.uniform((2, 3)))

# Tensor properties
randn_tf = tf.random.normal((2, 3))
print("Properties -> Shape:", randn_tf.shape, ", Dtype:", randn_tf.dtype, ", Device:", randn_tf.device)

