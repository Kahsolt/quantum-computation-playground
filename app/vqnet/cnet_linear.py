#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyvqnet.tensor import tensor, QTensor
from pyvqnet import nn, qnn
from pyvqnet.optim import Adam

sel = 1
if sel == 1:
  x = QTensor([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]).reshape((-1, 1))
  y = QTensor([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]).reshape((-1, 1))
elif sel == 2:
  x = tensor.arange(-10, 10).reshape((-1, 1))
  y = 2 * x - 1 + tensor.randn(x.shape)

x.requires_grad = True
print('x.shape:', x.shape)

model = nn.Linear(1, 1)
print(f'w: {model.weights[0]}, b: {model.bias[0]}')

optim = Adam(model.parameters(), lr=0.01)
creterion = nn.MeanSquaredError()

for i in range(1000):
  y_hat = model(x)

  loss = creterion(y, y_hat)
  optim.zero_grad()
  loss.backward()
  optim._step()

  if i % 10 == 0:
    print(f'loss: {loss.item():.7f}, w: {model.weights[0]}, b: {model.bias[0]}')
