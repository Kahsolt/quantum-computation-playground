#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import matplotlib.pyplot as plt
from pyvqnet import tensor, nn
from pyvqnet.optim import (
  SGD,
  Adagrad,
  Adadelta,
  RMSProp,
  Adam,
  Adamax,
)


# y = 2x - 1
x = tensor.arange(-10, 10).reshape((-1, 1))
y = 2 * x - 1 + tensor.randn(x.shape) * 0.5

x.requires_grad = True
print('x.shape:', x.shape)


LR = 0.01
OPTIMS = [
  SGD,
  Adagrad,
  Adadelta,
  RMSProp,
  Adam,
  Adamax,
]

creterion = nn.MeanSquaredError()
for optim_cls in OPTIMS:
  print(f'[{optim_cls.__name__}]')

  # starts form y = x
  model = nn.Linear(1, 1)
  #model.weights[0][0] = 1.0
  #model.bias   [0]    = 0.0
  optim = optim_cls(model.parameters(), lr=LR)

  losses = []
  for i in range(500):
    y_hat = model(x)

    loss = creterion(y, y_hat)
    optim.zero_grad()
    loss.backward()
    optim._step()

    if i % 10 == 0:
      #print(f'loss: {loss.item():.7f}, w: {model.weights[0]}, b: {model.bias[0]}')
      losses.append(loss.item())

  print(f'w: {model.weights[0]}, b: {model.bias[0]}')
  plt.plot(losses, label=optim_cls.__name__)

plt.legend()
plt.show()
