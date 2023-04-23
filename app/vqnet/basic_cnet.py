#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyvqnet.tensor import tensor, QTensor
from pyvqnet import nn, qnn
from pyvqnet.optim import Adam

ic = 3
oc = 2
b  = 2
hw = 6

x0 = tensor.arange(1, b * ic * hw * hw + 1)
x0 = x0.reshape([b, ic, hw, hw])
x0.requires_grad = True

model = nn.Conv2D(ic, oc, (3, 3), (2, 2), 'same')

y = model(x0)
y = nn.Sigmoid()(y)
print(y)

optim = Adam(model.parameters(), lr=0.005)
creterion = nn.BinaryCrossEntropy()

z = tensor.full_like(y, 0.5)
loss = creterion(z, y)
loss.backward()

print(model.weights.grad)
print(model.bias.grad)
print(x0.grad)
print(y)
