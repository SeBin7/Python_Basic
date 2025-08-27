"""
encapsulation.py
Encapsulation in Python OOP: public/protected/private, name-mangling, @property,
validation, __slots__, and framework-style encapsulation with PyTorch / TensorFlow.
"""

# =====================================
# 1) Python Encapsulation Basics
# =====================================
class BankAccount:
    # NOTE: Python has conventions, not true access modifiers.
    # - public:    balance (via @property), deposit(), withdraw()
    # - protected: _account_no (convention: single underscore)
    # - private:   __balance (name-mangled -> _BankAccount__balance)
    def __init__(self, owner: str, account_no: str, balance: float = 0.0):
        self.owner = owner                  # public
        self._account_no = account_no       # "protected" by convention
        self.__balance = float(balance)     # private (name-mangled)
        self._interest_rate = 0.01          # managed via property

    # read-only property for balance
    @property
    def balance(self) -> float:
        return self.__balance

    # validated property for interest rate
    @property
    def interest_rate(self) -> float:
        return self._interest_rate

    @interest_rate.setter
    def interest_rate(self, r: float):
        if not (0.0 <= r <= 0.2):
            raise ValueError("interest_rate must be between 0.0 and 0.2")
        self._interest_rate = float(r)

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("deposit amount must be positive")
        self.__balance += amount

    def withdraw(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("withdraw amount must be positive")
        if amount > self.__balance:
            raise ValueError("insufficient funds")
        self.__balance -= amount

    def apply_interest(self):
        self.__balance *= (1.0 + self._interest_rate)

acc = BankAccount("Alice", "001-123", 100.0)
print("owner:", acc.owner)                      # owner: Alice
print("account_no (protected):", acc._account_no)   # account_no (protected): 001-123
print("balance (via property):", acc.balance)   # balance (via property): 100.0

acc.deposit(50)
print("after deposit:", acc.balance)            # after deposit: 150.0
acc.withdraw(20)
print("after withdraw:", acc.balance)           # after withdraw: 130.0
acc.interest_rate = 0.05
acc.apply_interest()
print("after interest:", acc.balance)           # after interest: 136.5

# Name-mangling demonstration
print(hasattr(acc, "__balance"))                # False
print(hasattr(acc, "_BankAccount__balance"))    # True

# =====================================
# 2) __slots__ to restrict dynamic attributes (memory + safety)
# =====================================
class Point2D:
    __slots__ = ("x", "y")  # prevent creating attributes other than x, y
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y

p = Point2D(1.0, 2.0)
print("point:", p.x, p.y)                       # point: 1.0 2.0
try:
    p.z = 3.0  # this will fail because of __slots__
except AttributeError as e:
    print("slots error:", type(e).__name__)     # slots error: AttributeError


# =====================================
# 3) PyTorch: encapsulating config with @property + validation
# =====================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self._dropout = nn.Dropout(dropout_p)
        self._trainable = True  # encapsulated trainable flag

    # Encapsulate dropout probability with validation
    @property
    def dropout_p(self) -> float:
        return float(self._dropout.p)

    @dropout_p.setter
    def dropout_p(self, p: float):
        if not (0.0 <= p < 1.0):
            raise ValueError("dropout_p must be in [0.0, 1.0)")
        self._dropout.p = float(p)

    # Encapsulate requires_grad toggling for all parameters
    @property
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    def trainable(self, flag: bool):
        self._trainable = bool(flag)
        for p in self.parameters():
            p.requires_grad = self._trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = F.relu(x)
        x = self._dropout(x)
        return x

block = MyBlock(4, 2, dropout_p=0.0)
print("PT initial dropout_p:", block.dropout_p)  # PT initial dropout_p: 0.0
block.dropout_p = 0.3
print("PT updated dropout_p:", block.dropout_p)  # PT updated dropout_p: 0.3

x = torch.randn(1, 4)
y = block(x)
print("PT output shape:", y.shape)               # PT output shape: torch.Size([1, 2])

block.trainable = False
print("PT trainable now:", block.trainable)      # PT trainable now: False
print(all(p.requires_grad for p in block.parameters()))  # False


# =====================================
# 4) TensorFlow: encapsulating config in tf.keras.Model
# =====================================
import tensorflow as tf

# Optional: safer GPU memory usage on small GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("TF memory growth setup failed:", e)

class TFBlock(tf.keras.Model):
    def __init__(self, out_dim: int, dropout_rate: float = 0.0):
        super().__init__()
        self.dense = tf.keras.layers.Dense(out_dim, activation="relu")
        self._dropout = tf.keras.layers.Dropout(dropout_rate)
        self._trainable_flag = True

    # Encapsulate dropout rate with validation
    @property
    def dropout_rate(self) -> float:
        return float(self._dropout.rate)

    @dropout_rate.setter
    def dropout_rate(self, r: float):
        if not (0.0 <= r < 1.0):
            raise ValueError("dropout_rate must be in [0.0, 1.0)")
        # Keras Dropout exposes .rate; updating is acceptable for simple demos
        self._dropout.rate = float(r)

    # Encapsulate model trainable flag
    @property
    def trainable_flag(self) -> bool:
        return self._trainable_flag

    @trainable_flag.setter
    def trainable_flag(self, flag: bool):
        self._trainable_flag = bool(flag)
        self.trainable = self._trainable_flag  # standard Keras switch

    def call(self, x, training=False):
        x = self.dense(x)
        x = self._dropout(x, training=training)
        return x

tf_block = TFBlock(out_dim=2, dropout_rate=0.0)
print("TF initial dropout_rate:", tf_block.dropout_rate)   # TF initial dropout_rate: 0.0
tf_block.dropout_rate = 0.4
print("TF updated dropout_rate:", tf_block.dropout_rate)   # TF updated dropout_rate: 0.4

x_tf = tf.random.normal((1, 4))
y_tf = tf_block(x_tf, training=True)
print("TF output shape:", y_tf.shape)                      # TF output shape: (1, 2)

tf_block.trainable_flag = False
print("TF trainable now:", tf_block.trainable_flag)        # TF trainable now: False
