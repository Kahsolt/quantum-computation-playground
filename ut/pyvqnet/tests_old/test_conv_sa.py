#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

from pyvqnet import tensor
from pyvqnet import nn
from utils import model_inspect

sa = nn.Self_Conv_Attention(3, 16)    # in_dim, attn_dim
model_inspect(sa)

x = tensor.randn([1, 3, 32, 32])
print('x.shape:', x.shape)
y, m = sa(x)
print('y.shape:', y.shape)
print('m.shape:', m.shape)
