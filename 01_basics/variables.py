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

print("Values:", integer_number, floating_number, message, flag)  # Values: 10 3.14 Hello True

# Check variable types
print("Types:", type(integer_number), type(floating_number), type(message), type(flag))  # <class 'int'> <class 'float'> <class 'str'> <class 'bool'>

# Examples
x, y, z = 10, "Hello", 3.14   # Multiple assignment
p = q = r = 100               # Same value assignment
m, n = 5, 10
m, n = n, m                   # Swapping

print("Assignments:", x, y, z)  # Assignments: 10 Hello 3.14
print("Same value:", p, q, r)   # Same value: 100 100 100
print("Swapped:", m, n)         # Swapped: 10 5

# Dynamic typing
var = 42
print("Dynamic typing (int):", var, type(var))  # 42 <class 'int'>
var = "text"
print("Dynamic typing (str):", var, type(var))  # text <class 'str'>

# Constants convention (by naming convention only)
PI = 3.14159
MAX_USERS = 100
print("Constants:", PI, MAX_USERS)  # Constants: 3.14159 100

# Reference and id()
a = 10
b = a
print("id(a):", id(a), "id(b):", id(b), "a is b:", a is b)  # a is b: True

# Mutable vs Immutable
num1 = 5
num2 = num1
num2 += 1
print("Immutable (int):", num1, num2)  # Immutable (int): 5 6

lst1 = [1, 2, 3]
lst2 = lst1
lst2.append(4)
print("Mutable (list):", lst1, lst2)  # Mutable (list): [1, 2, 3, 4] [1, 2, 3, 4]

# Type casting
num_str = "123"
num_int = int(num_str)
num_float = float(num_str)
print("Casting:", num_int, num_float, str(3.14))  # Casting: 123 123.0 '3.14'

# None type
nothing = None
print("None value:", nothing, type(nothing))  # None <class 'NoneType'>

# Underscore variable
for _ in range(3):
    print("Repeat without using index")  # printed 3 times

# Equality vs Identity
list_a = [1, 2, 3]
list_b = [1, 2, 3]
print("list_a == list_b:", list_a == list_b) # True (values equal)
print("list_a is list_b:", list_a is list_b) # False (different objects)


big_x = 1000
big_y = 1000
print("big_x == big_y:", big_x == big_y) # True
print("big_x is big_y:", big_x is big_y) # False (different int objects, usually)


small_x = 10
small_y = 10
print("small_x is small_y:", small_x is small_y) # True (cached small int)


short_s1 = "hello"
short_s2 = "hello"
print("short_s1 is short_s2:", short_s1 is short_s2) # True (interned string)


long_s1 = "hello world! this is long"
long_s2 = "hello world! this is long"
print("long_s1 is long_s2:", long_s1 is long_s2) # May be True (interning optimization)

# Literal vs runtime string creation
literal_str1 = "abc" * 10000
literal_str2 = "abc" * 10000
print("literal_str1 is literal_str2:", literal_str1 is literal_str2) # False (created at runtime)


literal_same1 = "abcdefghijabcdefghij"
literal_same2 = "abcdefghijabcdefghij"
print("literal_same1 is literal_same2:", literal_same1 is literal_same2) # May be True (interned)


# =====================================
# 2. PyTorch Tensors
# =====================================
print("\n=== PyTorch Tensors ===")

# Basic tensors
scalar_t = torch.tensor(5)
vector_t = torch.tensor([1, 2, 3])
matrix_t = torch.tensor([[1, 2], [3, 4]])

print("Scalar:", scalar_t)   # tensor(5)
print("Vector:", vector_t)   # tensor([1, 2, 3])
print("Matrix:\n", matrix_t)  # tensor([[1, 2],[3, 4]])

# Predefined tensors
print("Ones:\n", torch.ones(2, 3))   # 2x3 tensor of ones
print("Zeros:\n", torch.zeros(2, 3)) # 2x3 tensor of zeros
print("Full:\n", torch.full((2, 3), 7)) # 2x3 tensor filled with 7

# Random tensors
print("Randn (Normal):\n", torch.randn(2, 3))   # random values from normal distribution
print("Rand (Uniform):\n", torch.rand(2, 3))    # random values from uniform distribution [0,1)

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

print("Scalar:", scalar_tf)   # tf.Tensor(5, shape=(), dtype=int32)
print("Vector:", vector_tf)   # tf.Tensor([1 2 3], shape=(3,), dtype=int32)
print("Matrix:\n", matrix_tf) # tf.Tensor([[1 2][3 4]], shape=(2,2), dtype=int32)

# Predefined tensors
print("Ones:\n", tf.ones((2, 3)))   # 2x3 tensor of ones
print("Zeros:\n", tf.zeros((2, 3))) # 2x3 tensor of zeros
print("Full:\n", tf.fill((2, 3), 7)) # 2x3 tensor filled with 7

# Random tensors
print("Normal:\n", tf.random.normal((2, 3)))   # random values from normal distribution
print("Uniform:\n", tf.random.uniform((2, 3))) # random values from uniform distribution [0,1)

# Tensor properties
randn_tf = tf.random.normal((2, 3))
print("Properties -> Shape:", randn_tf.shape, ", Dtype:", randn_tf.dtype, ", Device:", randn_tf.device)
