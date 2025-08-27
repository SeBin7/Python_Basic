"""
polymorphism.py
Show polymorphism patterns:
- Python: subclassing-based, duck typing, ABC interface, singledispatch (ad-hoc)
- PyTorch: different nn.Module models share same interface (forward)
- TensorFlow: different tf.keras.Model models share same interface (call)
"""

# =====================================
# 1) Python polymorphism: subclassing
# =====================================
class Animal:
    def speak(self) -> str:
        return "..."  # default behavior

class Dog(Animal):
    def speak(self) -> str:
        return "Woof!"

class Cat(Animal):
    def speak(self) -> str:
        return "Meow!"

def let_it_speak(animal: Animal) -> str:
    # subtype polymorphism: any subclass of Animal works
    return animal.speak()

pets = [Dog(), Cat(), Animal()]
for p in pets:
    print(type(p).__name__, "->", let_it_speak(p))
    # Dog -> Woof!
    # Cat -> Meow!
    # Animal -> ...

# =====================================
# 2) Python polymorphism: duck typing (no inheritance needed)
# =====================================
class Robot:
    def speak(self) -> str:
        return "Beep!"

def talk(obj) -> str:
    # duck typing: if it has .speak(), it works
    return obj.speak()

print("Robot ->", talk(Robot()))  # Robot -> Beep!
print("Dog   ->", talk(Dog()))    # Dog   -> Woof!

# =====================================
# 3) Python polymorphism: ABC-enforced interface
# =====================================
from abc import ABC, abstractmethod
import math

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        ...

class Rectangle(Shape):
    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
    def area(self) -> float:
        return self.w * self.h

class Circle(Shape):
    def __init__(self, r: float):
        self.r = r
    def area(self) -> float:
        return math.pi * self.r * self.r

def total_area(shapes: list[Shape]) -> float:
    return sum(s.area() for s in shapes)

shapes: list[Shape] = [Rectangle(3, 4), Circle(5)]
print("areas:", [round(s.area(), 2) for s in shapes])  # areas: [12, 78.54]
print("total_area:", round(total_area(shapes), 2))      # total_area: 90.54

# =====================================
# 4) Python polymorphism: ad-hoc via singledispatch
# =====================================
from functools import singledispatch

@singledispatch
def to_text(x) -> str:
    return f"<unknown:{type(x).__name__}>"

@to_text.register
def _(x: int) -> str:
    return f"<int:{x}>"

@to_text.register
def _(x: list) -> str:
    return f"<list:{len(x)} items>"

@to_text.register
def _(x: dict) -> str:
    return f"<dict:{list(x.keys())}>"

print(to_text(7))                 # <int:7>
print(to_text([1,2,3]))           # <list:3 items>
print(to_text({"a":1, "b":2}))    # <dict:['a', 'b']>
print(to_text(3.14))              # <unknown:float>

# =====================================
# 5) PyTorch polymorphism: different models, same interface (forward)
# =====================================
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)  # for reproducible shapes/paths (values not printed)

class NetA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # logits (N,2)

class NetB(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def run_torch_model(model: nn.Module, x: torch.Tensor) -> torch.Size:
    y = model(x)      # polymorphic call: same .forward contract
    print(type(model).__name__, "->", y.shape)
    return y.shape

x_t = torch.randn(1, 4)
run_torch_model(NetA(), x_t)  # NetA -> torch.Size([1, 2])
run_torch_model(NetB(), x_t)  # NetB -> torch.Size([1, 2])

# =====================================
# 6) TensorFlow polymorphism: different models, same interface (call)
# =====================================
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("TF memory growth setup failed:", e)

tf.random.set_seed(0)

class TFNetA(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc = tf.keras.layers.Dense(2)  # (4->2)
    def call(self, x, training=False):
        return self.fc(x)  # logits (N,2)

class TFNetB(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(8, activation="relu")
        self.fc2 = tf.keras.layers.Dense(2)
    def call(self, x, training=False):
        x = self.fc1(x, training=training)
        return self.fc2(x, training=training)

def run_tf_model(model: tf.keras.Model, x: tf.Tensor) -> tf.TensorShape:
    y = model(x, training=False)  # polymorphic call: same .call contract
    print(type(model).__name__, "->", y.shape)
    return y.shape

x_tf = tf.random.normal((1, 4))
run_tf_model(TFNetA(), x_tf)  # TFNetA -> (1, 2)
run_tf_model(TFNetB(), x_tf)  # TFNetB -> (1, 2)
