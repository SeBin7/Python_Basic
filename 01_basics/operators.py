"""
operators.py
Examples of Python operators, PyTorch operators, and TensorFlow operators
"""

import torch
import tensorflow as tf

# =====================================
# 1. Python Operators
# =====================================

# Arithmetic 
a, b = 7, 3
print("[Python] Addition:", a + b)
print("[Python] Subtraction:", a - b)
print("[Python] Multiplication:", a * b)
print("[Python] Division:", a / b)
print("[Python] Floor Division:", a // b)
print("[Python] Modulus:", a % b)
print("[Python] Exponent:", a ** b)

# Comparison 
print("[Python] a > b:", a > b)
print("[Python] a == b:", a == b)
print("[Python] a != b:", a != b)

# Logical 
x, y = True, False
print("[Python] AND:", x and y)
print("[Python] OR:", x or y)
print("[Python] NOT:", not x)

# Assignment 
c = 10
c += 5
print("[Python] c after += 5:", c)

# Membership & Identity
nums = [1, 2, 3, 4]
print("[Python] 2 in nums:", 2 in nums)
print("[Python] 10 not in nums:", 10 not in nums)

p = [1, 2, 3]
q = p
r = [1, 2, 3]
print("[Python] p is q:", p is q)
print("[Python] p is r:", p is r)


# =====================================
# 2. PyTorch Operators
# =====================================

t1 = torch.tensor([7, 3])
t2 = torch.tensor([2, 4])

# Arithmetic
print("[PyTorch] add:", t1 + t2)
print("[PyTorch] mul:", t1 * t2)

# Comparison
print("[PyTorch] comparison:", t1 > t2)

# Logical
bool_t1 = torch.tensor([True, False])
bool_t2 = torch.tensor([False, True])
print("[PyTorch] logical AND:", bool_t1 & bool_t2)

# Assignment (in-place)
t = torch.tensor([1.0, 2.0, 3.0])
t.add_(5)
print("[PyTorch] in-place add:", t)


# =====================================
# 3. TensorFlow Operators
# =====================================

tf1 = tf.constant([7, 3])
tf2 = tf.constant([2, 4])

# Arithmetic
print("[TensorFlow] add:", tf1 + tf2)
print("[TensorFlow] mul:", tf1 * tf2)

# Comparison
print("[TensorFlow] comparison:", tf.greater(tf1, tf2))

# Logical
bool_tf1 = tf.constant([True, False])
bool_tf2 = tf.constant([False, True])
print("[TensorFlow] logical AND:", tf.logical_and(bool_tf1, bool_tf2))

# Assignment → TensorFlow는 immutable, 직접 연산 결과 새로 할당
tf_var = tf.Variable([1.0, 2.0, 3.0])
tf_var.assign_add([5.0, 5.0, 5.0])
print("[TensorFlow] assign_add:", tf_var)
