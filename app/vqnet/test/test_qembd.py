#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/21 

import pyqpanda as pq
from pyvqnet.tensor import tensor
from pyvqnet.qnn import QuantumLayerV2, Quantum_Embedding

from utils import train_dummy

# RX-RY-RZ (???)
#  para_cnt ~= 6 * D * n_repeat_input * n_unitary_layers

D = 4
n_repeat = 2
n_repeat_input = 2
n_unitary_layers = 2
nq = D * n_repeat_input

qvm = pq.CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(nq)

B = 16
X = tensor.randn([B, D])
Y = tensor.randn([B, 1])    # FIXME: any-dim to 1-dim, why?
X.requires_grad = True

qe = Quantum_Embedding(qubits, qvm, n_repeat_input, D, n_unitary_layers, n_repeat)
model = QuantumLayerV2(qe.compute_circuit, qe.param_num)  

train_dummy(model, X, Y)
