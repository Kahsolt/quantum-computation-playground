#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyqpanda import *
from pyvqnet.tensor import tensor, QTensor
from pyvqnet import nn, qnn
from pyvqnet.optim import Adam

def qcricuit(x, param, qubits, cubits, qvm):
  c = QCircuit()
  c << RZ(qubits[0], x[0])
  c << RZ(qubits[1], x[1])
  c << RZ(qubits[2], x[2])
  c << RZ(qubits[3], x[3])
  c << CNOT(qubits[0], qubits[1])
  c << RZ(qubits[1], param[0])
  c << CNOT(qubits[1], qubits[2])
  c << RZ(qubits[2], param[1])
  c << CNOT(qubits[2], qubits[3])
  c << RZ(qubits[3], param[2])

  prog = QProg() << c
  print(prog)
  
  rlt_prob = qnn.ProbsMeasure([0, 2], prog, qvm, qubits)
  return rlt_prob

param_num  = 3
num_qubits = 4
pqc = qnn.QuantumLayer(qcricuit, param_num, 'cpu', num_qubits)

x = QTensor([[1, 2, 3, 4]])
x.requires_grad = True
rlt = pqc(x)
rlt.backward()

print(x.shape)
print(x)

print(pqc.parameters())
