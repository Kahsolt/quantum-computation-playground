#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyvqnet.tensor import tensor, QTensor

a = tensor.ones([1, 2, 3, 4])
b = tensor.full([1, 2, 3, 4], 12)

print(dir(a))

print(tensor.reshape(a, [3, 8]).shape)
print(tensor.squeeze(a).shape)

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(tensor.mean(a, 1))
print(tensor.cos(a))
print(tensor.sqrt(a))
print(tensor.log(a))
print(tensor.greater(a, b))
print(tensor.logical_xor(a, b))
