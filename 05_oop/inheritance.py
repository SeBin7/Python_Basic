"""
inheritance.py (all-in-one)
- Python inheritance basics, super(), overriding
- Abstract base class (abc)
- MRO (Method Resolution Order)
- Multiple inheritance + Mixin
- PyTorch nn.Module inheritance
- TensorFlow tf.keras.Model inheritance
"""

# =====================================
# 0) Utilities (optional)
# =====================================
from __future__ import annotations


# =====================================
# 1) Python Inheritance Basics: super() & overriding
# =====================================
class Animal:
    def __init__(self, name: str):
        self.name = name  # base attribute

    def speak(self) -> str:
        return "..."  # generic sound

    def info(self) -> str:
        return f"Animal(name={self.name})"


class Dog(Animal):
    def __init__(self, name: str, breed: str):
        # cooperative init
        super().__init__(name)       # initialize parent-managed state
        self.breed = breed           # child-specific attribute

    # override behavior completely
    def speak(self) -> str:
        return "Woof!"

    # extend parent behavior via super()
    def info(self) -> str:
        base = super().info()        # e.g., "Animal(name=Rex)"
        return f"{base}, Dog(breed={self.breed})"


class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"


a = Animal("Generic")
d = Dog("Rex", "Border Collie")
c = Cat("Mimi")

print(a.speak())               # ...
print(d.speak())               # Woof!
print(c.speak())               # Meow!
print(d.info())                # Animal(name=Rex), Dog(breed=Border Collie)

print(isinstance(d, Animal))   # True
print(issubclass(Dog, Animal)) # True


# =====================================
# 2) Abstract Base Class (abc)
# =====================================
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        """Subclasses must implement area()."""
        ...

    @abstractmethod
    def perimeter(self) -> float:
        """Subclasses must implement perimeter()."""
        ...


class Rectangle(Shape):
    def __init__(self, w: float, h: float):
        self.w, self.h = w, h

    def area(self) -> float:
        return self.w * self.h

    def perimeter(self) -> float:
        return 2 * (self.w + self.h)


class Circle(Shape):
    def __init__(self, r: float):
        self.r = r

    def area(self) -> float:
        return math.pi * self.r * self.r

    def perimeter(self) -> float:
        return 2 * math.pi * self.r


rect = Rectangle(3, 4)
circle = Circle(5)
print("Rectangle area:", rect.area())              # Rectangle area: 12
print("Rectangle perimeter:", rect.perimeter())    # Rectangle perimeter: 14
print("Circle area:", round(circle.area(), 2))     # Circle area: 78.54
print("Circle perimeter:", round(circle.perimeter(), 2))  # Circle perimeter: 31.42
# Shape()  # TypeError: Can't instantiate abstract class Shape (uncomment to test)


# =====================================
# 3) MRO (Method Resolution Order) & Multiple Inheritance basics
# =====================================
class A:
    def greet(self):
        return "Hello from A"

class B(A):
    def greet(self):
        return "Hello from B"

class C(A):
    def greet(self):
        return "Hello from C"

class D(B, C):  # multiple inheritance
    pass

d_mro = D()
print(d_mro.greet())     # Hello from B (B precedes C in MRO)
print(D.__mro__)         # (<class '__main__.D'>, <class '__main__.B'>, <class '__main__.C'>, <class '__main__.A'>, <class 'object'>)


# =====================================
# 4) Multiple Inheritance + Mixin pattern
# =====================================
class JSONMixin:
    """Provide JSON serialization."""
    def to_json(self) -> str:
        import json
        # This naive approach dumps instance __dict__.
        return json.dumps(self.__dict__)

class PrintableMixin:
    """Provide a pretty string representation."""
    def __str__(self):
        return f"<{self.__class__.__name__} {self.__dict__}>"

class User(JSONMixin, PrintableMixin):
    def __init__(self, username: str, email: str):
        self.username = username
        self.email = email

u = User("alice", "alice@example.com")
print(u)            # <User {'username': 'alice', 'email': 'alice@example.com'}>
print(u.to_json())  # {"username": "alice", "email": "alice@example.com"}
print(User.__mro__) # (<class '__main__.User'>, <class '__main__.JSONMixin'>, <class '__main__.PrintableMixin'>, <class 'object'>)


# =====================================
# 5) PyTorch Inheritance (nn.Module)
# =====================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)  # registered layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base behavior
        return self.linear(x)

class ReLUBlock(BaseBlock):
    def __init__(self, in_dim: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__(in_dim, out_dim)         # cooperative init
        self.dropout = nn.Dropout(dropout_p)

    # override forward and extend behavior
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)                    # call parent forward (Linear)
        x = F.relu(x)                             # add non-linearity
        x = self.dropout(x)                       # optional dropout
        return x

block = ReLUBlock(4, 2, dropout_p=0.0)
x_torch = torch.randn(1, 4)
y_torch = block(x_torch)
print("PyTorch output shape:", y_torch.shape)  # PyTorch output shape: torch.Size([1, 2])


# =====================================
# 6) TensorFlow Inheritance (tf.keras.Model)
# =====================================
import tensorflow as tf

# Safer GPU memory usage on small cards (optional)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("TF memory growth setup failed:", e)

class BaseDense(tf.keras.Model):
    def __init__(self, out_dim: int):
        super().__init__()
        self.fc = tf.keras.layers.Dense(out_dim)

    def call(self, x, training=False):
        return self.fc(x)

class GELUDense(BaseDense):
    def __init__(self, out_dim: int, dropout_p: float = 0.0):
        super().__init__(out_dim)                     # cooperative init
        self.dropout = tf.keras.layers.Dropout(dropout_p)

    # override call and extend behavior
    def call(self, x, training=False):
        x = super().call(x, training=training)        # parent call (Dense)
        x = tf.nn.gelu(x)                              # add activation
        x = self.dropout(x, training=training)         # conditional dropout
        return x

tf_model = GELUDense(out_dim=2, dropout_p=0.0)
x_tf = tf.random.normal((1, 4))
y_tf = tf_model(x_tf, training=False)
print("TensorFlow output shape:", y_tf.shape)         # TensorFlow output shape: (1, 2)
