#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

from pyvqnet import tensor
from pyvqnet.qnn.qlinear import QLinear
from utils import train_dummy

# X-H + RY-CZ pair-wise cycle
# NOTE: it has no trainable params, is it broken?

# NOTE: in_ch and out_ch must >= 2, due to its pair-wise implementation
B = 8
I = 6
O = 2

x = tensor.randn([B, I])
y = tensor.randn([B, O])
model = QLinear(I, O)
train_dummy(model, x, y)
