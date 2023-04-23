#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import pyqpanda as pq
from pyvqnet.qnn import grad, ProbsMeasure

# approximate grad by limited differential?

def pqctest(param):
  qvm = pq.CPUQVM()
  qvm.init_qvm()
  qv = qvm.qAlloc_many(2)

  qc = pq.QCircuit() \
     << pq.RX(qv[0], param[0]) \
     << pq.RY(qv[1], param[1]) \
     << pq.CNOT(qv[0], qv[1]) \
     << pq.RX(qv[1], param[2])
  prog = pq.QProg() << qc

  return ProbsMeasure([1], prog, qvm, qv)


g = grad(pqctest, [0.1, 0.2, 0.3])
print(g)
exp = pqctest([0.1, 0.2, 0.3])
print(exp)

# what the fuck?
print(grad(lambda x: x * 0.5, [0.1]))
print(grad(lambda x: x * 1.5, [0.1]))
print(grad(lambda x: x * 2.5, [0.1]))
print(grad(lambda x: x * 3.5, [0.1]))
print(grad(lambda x: x * 4.5, [0.1]))
