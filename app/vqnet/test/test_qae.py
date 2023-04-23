#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import numpy as np
from pyvqnet.qnn.qae import QAElayer
from pyvqnet.nn.loss import fidelityLoss

B = 8
C = 1
H = 2
W = 2

x = np.random.uniform(size=[B, C, H, W])

encode_qubits = 4
assert encode_qubits == H * W
latent_qubits = 2
trash_qubits = encode_qubits - latent_qubits
total_qubits = 1 + trash_qubits + encode_qubits
print('trash_qubits:', trash_qubits)
print('total_qubits:', total_qubits)
model = QAElayer(trash_qubits, total_qubits)

model.history_prob
model.merge_opinfo
model.machine
model.clist
model.qlist
model.n_qubits
model.n_aux_qubits
model.n_trash_qubits

x = x.reshape((B, C*H*W))
x = np.concatenate((np.zeros([B, 1 + trash_qubits]), x), 1)
print('x.shape:', x.shape)
y = model(x)
print('y.shape:', y.shape)
print(y)

l = fidelityLoss()(y)   # => just 1/y
print(l)
