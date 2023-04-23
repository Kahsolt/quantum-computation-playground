#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import matplotlib.pyplot as plt
from pyqpanda import *
from pyvqnet.tensor import QTensor
from pyvqnet.optim import Rotosolve
from pyvqnet.qnn.measure import expval

qvm = CPUQVM()
qvm.init_qvm()
qv = qvm.qAlloc_many(2)

def gen(param, generators, qbits, circuit):
  if   generators == 'X': circuit.insert(RX(qbits, param))
  elif generators == 'Y': circuit.insert(RY(qbits, param))
  else:                   circuit.insert(RZ(qbits, param))

def circuits(params, generators, circuit):
  gen(params[0], generators[0], qv[0], circuit)
  gen(params[1], generators[1], qv[1], circuit)
  return QProg() << circuit << CNOT(qv[0], qv[1])

def ansatz1(params:QTensor, generators):
  prog = circuits(params.getdata(), generators, QCircuit())
  return expval(qvm, prog, {'Z0': 1}, qv), expval(qvm, prog, {'Y1': 1}, qv)

def ansatz2(params:QTensor, generators):
  prog = circuits(params.getdata(), generators, QCircuit())
  return expval(qvm, prog, {'X0' : 1}, qv)

def loss(params):
  Z, Y = ansatz1(params, ['X', 'Y'])
  X    = ansatz2(params, ['X', 'Y'])
  return 0.5 * Y + 0.8 * Z - 0.2 * X

t = QTensor([0.3, 0.25])
opt = Rotosolve(max_iter=50)
costs_rotosolve = opt.minimize(t, loss)
print(type(costs_rotosolve))
print(len(costs_rotosolve))
print(costs_rotosolve)

plt.plot(costs_rotosolve, 'o-')
plt.title('rotosolve')
plt.xlabel('cycles')
plt.ylabel('cost')
plt.show()
