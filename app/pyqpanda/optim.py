#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/28 

from pyqpanda import *
import numpy as np
import matplotlib.pyplot as plt


OPTIMIZERS = [
  lambda loss: VanillaGradientDescentOptimizer.minimize(loss, 0.01, 1.e-6),
  lambda loss: MomentumOptimizer.minimize(loss, 0.01, 0.9),
  lambda loss: AdaGradOptimizer.minimize(loss, 0.01, 0.9, 1.e-10),
  lambda loss: RMSPropOptimizer.minimize(loss, 0.01, 0.9, 1.e-10),
  lambda loss: AdamOptimizer.minimize(loss, 0.01, 0.9, 0.999, 1.e-10),
]

for optim in OPTIMIZERS:
  # linear
  x = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
  y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
  X = var(x.reshape(len(x), 1))
  Y = var(y.reshape(len(y), 1))

  W = var(0, True)
  B = var(0, True)
  Y_hat = W * X + B

  loss_fn = sum(poly(Y_hat - Y, var(2)) / len(x))
  optimizer = optim(loss_fn)
  leaves = optimizer.get_variables()

  for i in range(1000):
    optimizer.run(leaves, 0)
    loss = optimizer.get_loss()

  w2 = W.get_value()[0, 0]
  b2 = B.get_value()[0, 0]
  print("loss: ", loss, " W: ", w2, " b: ", w2)

  continue
  plt.plot(x, y, 'o', label = 'Training data')
  plt.plot(x, w2*x + b2, 'r', label = 'Fitted line')
  plt.legend()
  plt.show()
