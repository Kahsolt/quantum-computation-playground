#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyvqnet.tensor import QTensor
from pyvqnet.nn.loss import BinaryCrossEntropy, CategoricalCrossEntropy, SoftmaxCrossEntropy, CrossEntropyLoss, NLL_Loss

CRETERIONS = [
#  BinaryCrossEntropy,         # onehot
#  SoftmaxCrossEntropy,        # onehot
#  CategoricalCrossEntropy,    # onehot, use SoftmaxCrossEntropy instead (deprecated)
  CrossEntropyLoss,            # integer, alike SoftmaxCrossEntropy
#  NLL_Loss,
]

# BCE requires all inputs in range [0, 1]
x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]])
y = QTensor([[0, 1, 0], [0, 0, 1]])

x = QTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]])

print('===== [Y as onehot] =====')

print('x.shape:', x.shape)
print('y.shape:', y.shape)

for creterion in CRETERIONS:
  print(creterion)

  loss_fn = creterion()
  result = loss_fn(y, x)
  print(result)


print('===== [Y as integer] =====')

y = y.argmax([-1], False)
print('x.shape:', x.shape)
print('y.shape:', y.shape)

for creterion in CRETERIONS:
  print(creterion)

  loss_fn = creterion()
  result = loss_fn(y, x)
  print(result)
