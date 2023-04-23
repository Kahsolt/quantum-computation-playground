#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import matplotlib.pyplot as plt
import numpy as np
from pyvqnet.tensor import QTensor
from pyvqnet.nn.activation import (
  ReLu,
  LeakyReLu,
  ELU,
  Tanh,
  Softmax,
  Softplus,
  Softsign,
  Sigmoid,
  HardSigmoid,
)


ACTS = [
  ReLu,
  LeakyReLu,
  ELU,
  Tanh,
  Softmax,
  Softplus,
  Softsign,
  Sigmoid,
  HardSigmoid,
]

x = QTensor(np.linspace(-5, 5, 100))
for act_cls in ACTS:
  plt.plot(x, act_cls()(x), label=act_cls.__name__)

plt.legend()
plt.show()
