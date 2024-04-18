#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import matplotlib.pyplot as plt

from pyvqnet.nn import MeanSquaredError
from pyvqnet.optim import Adam


def model_inspect(model):
  print('[modules]')
  for m in model.modules():
    print(f'  {type(m).__name__}')
  print(f'[parameters] (count: {sum([p.size for p in model.parameters()])})')
  for p in model.parameters():
    print(f'  {p.shape}')


def train_dummy(model, x, y):
  print('x.shape:', x.shape)
  print('y.shape:', y.shape)
  model_inspect(model)

  x.requires_grad = True
  model.train()

  optim = Adam(model.parameters(), lr=0.01)
  creterion = MeanSquaredError()

  losses = []
  for _ in range(100):
    y_hat = model(x)

    loss = creterion(y, y_hat)
    optim.zero_grad()
    loss.backward()
    optim._step()

    print(f'loss: {loss.item():.7f}')
    losses.append(loss.item())

  plt.plot(losses)
  plt.show()
