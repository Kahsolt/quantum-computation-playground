#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

from pyvqnet import tensor
from pyvqnet.qnn.qcnn import QConv

from utils import train_dummy

# RY-RZ + CZ cycle + vU3
#   para_cnt: 36 = 3(n_channel) * 4(kernel_size) * 3(n_param_per_U3_gate)
# NOTE: 有概率遭遇loss一直上升的情况，应该是随机初始化不太行，应尝试重启

B = 8
H = 16
W = 16
I = 3
O = 1

x = tensor.randn([B, I, H,    W])
y = tensor.randn([B, O, H//2, W//2])
model = QConv(I, O, 4, stride=(2, 2))
train_dummy(model, x, y)
