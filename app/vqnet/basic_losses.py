#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyvqnet.tensor import tensor, QTensor
from pyvqnet.nn.loss import BinaryCrossEntropy, CategoricalCrossEntropy, SoftmaxCrossEntropy, CrossEntropyLoss, NLL_Loss

CRETERIONS = [
  BinaryCrossEntropy, 
  CategoricalCrossEntropy, 
  SoftmaxCrossEntropy, 
  CrossEntropyLoss,
  NLL_Loss,
]

x = QTensor([[0.3, 0.7, 0.2], [0.2, 0.3, 0.1]], requires_grad=True)
y = QTensor([[0, 1, 0], [0, 0, 1]], requires_grad=True)

#x = QTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]], requires_grad=True)
#y = QTensor([[0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], requires_grad=True)

for creterion in CRETERIONS:
  loss_fn = creterion()
  result = loss_fn(y, x)
  print(result)
