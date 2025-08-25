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
print("[Python] Addition:", a + b) # 10
print("[Python] Subtraction:", a - b) # 4
print("[Python] Multiplication:", a * b) # 21
print("[Python] Division:", a / b) # 2.333...
print("[Python] Floor Division:", a // b) # 2
print("[Python] Modulus:", a % b) # 1
print("[Python] Exponent:", a ** b) # 343


# Comparison
print("[Python] a > b:", a > b) # True
print("[Python] a == b:", a == b) # False
print("[Python] a != b:", a != b) # True


# Logical
x, y = True, False
print("[Python] AND:", x and y) # False
print("[Python] OR:", x or y) # True
print("[Python] NOT:", not x) # False


# Assignment
c = 10
c += 5
print("[Python] c after += 5:", c) # 15


# Membership & Identity
nums = [1, 2, 3, 4]
print("[Python] 2 in nums:", 2 in nums) # True
print("[Python] 10 not in nums:", 10 not in nums) # True


p = [1, 2, 3]
q = p
r = [1, 2, 3]
print("[Python] p is q:", p is q) # True (same object)
print("[Python] p is r:", p is r) # False (different objects)


# Bitwise operators
bit_a, bit_b = 5, 3 # 0b0101 and 0b0011
print("[Python] Bitwise AND:", bit_a & bit_b) # 1
print("[Python] Bitwise OR:", bit_a | bit_b) # 7
print("[Python] Bitwise XOR:", bit_a ^ bit_b) # 6
print("[Python] Bitwise NOT:", ~bit_a) # -6
print("[Python] Left Shift:", bit_a << 1) # 10
print("[Python] Right Shift:", bit_a >> 1) # 2


# Conditional (ternary) operator
x, y = 10, 20
max_val = x if x > y else y
print("[Python] Max value:", max_val) # 20


# Chained comparison
n = 5
print("[Python] 1 < n < 10:", 1 < n < 10) # True


# Operator precedence
print("[Python] True or False and False:", True or False and False) # True
print("[Python] (True or False) and False:", (True or False) and False) # False


# Equality vs Identity reminder
lst1 = [1, 2, 3]
lst2 = [1, 2, 3]
lst3 = lst1
print("[Python] lst1 == lst2:", lst1 == lst2) # True (same values)
print("[Python] lst1 is lst2:", lst1 is lst2) # False (different objects)

print("[Python] lst1 == lst2:", lst1 == lst3) # True (same values)
print("[Python] lst1 is lst2:", lst1 is lst3) # True (same objects reference)


# =====================================
# 2. PyTorch Operators
# =====================================


t1 = torch.tensor([7, 3])
t2 = torch.tensor([2, 4])


# Arithmetic
print("[PyTorch] add:", t1 + t2) # tensor([9, 7])
print("[PyTorch] mul:", t1 * t2) # tensor([14, 12])


# Comparison
print("[PyTorch] comparison:", t1 > t2) # tensor([True, False])


# Logical
bool_t1 = torch.tensor([True, False])
bool_t2 = torch.tensor([False, True])
print("[PyTorch] logical AND:", bool_t1 & bool_t2) # tensor([False, False])
print("[PyTorch] logical OR:", bool_t1 | bool_t2) # tensor([True, True])


# Bitwise
t3 = torch.tensor([5, 3])
t4 = torch.tensor([1, 7])
print("[PyTorch] bitwise AND:", t3 & t4) # tensor([1, 3])
print("[PyTorch] bitwise OR:", t3 | t4) # tensor([5, 7])
print("[PyTorch] bitwise XOR:", t3 ^ t4) # tensor([4, 4])


# In-place assignment
t = torch.tensor([1.0, 2.0, 3.0])
t.add_(5)
print("[PyTorch] in-place add:", t) # tensor([6., 7., 8.])


# Chained comparison equivalent in PyTorch (element-wise)
print("[PyTorch] (t1 > 2) & (t1 < 10):", (t1 > 2) & (t1 < 10)) # tensor([ True, True])




# =====================================
# 3. TensorFlow Operators
# =====================================


tf1 = tf.constant([7, 3])
tf2 = tf.constant([2, 4])


# Arithmetic
print("[TensorFlow] add:", tf1 + tf2) # tf.Tensor([9 7], shape=(2,), dtype=int32)
print("[TensorFlow] mul:", tf1 * tf2) # tf.Tensor([14 12], shape=(2,), dtype=int32)


# Comparison
print("[TensorFlow] comparison:", tf.greater(tf1, tf2)) # tf.Tensor([ True False], shape=(2,), dtype=bool)


# Logical
bool_tf1 = tf.constant([True, False])
bool_tf2 = tf.constant([False, True])
print("[TensorFlow] logical AND:", tf.logical_and(bool_tf1, bool_tf2)) # tf.Tensor([False False], shape=(2,), dtype=bool)
print("[TensorFlow] logical OR:", tf.logical_or(bool_tf1, bool_tf2)) # tf.Tensor([ True True], shape=(2,), dtype=bool)


# Bitwise
tf3 = tf.constant([5, 3])
tf4 = tf.constant([1, 7])
print("[TensorFlow] bitwise AND:", tf.bitwise.bitwise_and(tf3, tf4)) # tf.Tensor([1 3], shape=(2,), dtype=int32)
print("[TensorFlow] bitwise OR:", tf.bitwise.bitwise_or(tf3, tf4)) # tf.Tensor([5 7], shape=(2,), dtype=int32)
print("[TensorFlow] bitwise XOR:", tf.bitwise.bitwise_xor(tf3, tf4)) # tf.Tensor([4 4], shape=(2,), dtype=int32)


# Chained comparison equivalent in TensorFlow (element-wise)
print("[TensorFlow] (tf1 > 2) & (tf1 < 10):", tf.logical_and(tf1 > 2, tf1 < 10)) # tf.Tensor([ True True], shape=(2,), dtype=bool)


# Assignment → TensorFlow is immutable, must use assign_add
tf_var = tf.Variable([1.0, 2.0, 3.0])
tf_var.assign_add([5.0, 5.0, 5.0])
print("[TensorFlow] assign_add:", tf_var) # tf.Tensor([6. 7. 8.], shape=(3,), dtype=float32)