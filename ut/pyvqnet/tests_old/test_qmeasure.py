#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/23 

import pyqpanda as pq
from pyvqnet.qnn.measure import (
  expval,
  QuantumMeasure,
  ProbsMeasure,
  DensityMatrixFromQstate,
  Mutal_Info,
  VN_Entropy,
  VarMeasure,
  Purity,
)

assert expval
assert QuantumMeasure
assert ProbsMeasure
assert DensityMatrixFromQstate
assert Mutal_Info
assert VN_Entropy
assert VarMeasure
assert Purity

input = [0.56, 0.1]
qvm = pq.init_quantum_machine(pq.QMachineType.CPU)
qv = qvm.qAlloc_many(3)

''' QuantumMeasure '''
qc = pq.QCircuit() \
   << pq.RZ(qv[0], input[0]) \
   << pq.CNOT(qv[0], qv[1]) \
   << pq.RY(qv[1], input[1]) \
   << pq.CNOT(qv[0], qv[2]) \
   << pq.H(qv[0]) \
   << pq.H(qv[1]) \
   << pq.H(qv[2])
prog = pq.QProg() << qc
measure_qubits = [0, 2]
rlt_quant = QuantumMeasure(measure_qubits, prog, qvm, qv)
print('QuantumMeasure:', rlt_quant)  # => [234, 247, 246, 273]

''' ProbsMeasure '''
qc = pq.QCircuit() \
   << pq.RZ(qv[0], input[0]) \
   << pq.CNOT(qv[0], qv[1]) \
   << pq.RY(qv[1], input[1]) \
   << pq.CNOT(qv[0], qv[2]) \
   << pq.H(qv[0]) \
   << pq.H(qv[1]) \
   << pq.H(qv[2])
prog = pq.QProg() << qc
measure_qubits = [0, 2]
rlt_prob = ProbsMeasure(measure_qubits, prog, qvm, qv)
print('ProbsMeasure:', rlt_prob)   # => [0.25, 0.25, 0.25, 0.25]

''' VarMeasure '''
qc = pq.QCircuit() \
   << pq.RX(qv[0], 1.2) \
   << pq.CNOT(qv[0], qv[1])
prog = pq.QProg() << qc
var_result = VarMeasure(qvm, prog, qv[0])    # only accecpt single qubit
print('VarMeasure:', var_result)

''' expval '''
qc = pq.QCircuit() \
   << pq.RZ(qv[0], input[0]) \
   << pq.CNOT(qv[0], qv[1]) \
   << pq.RY(qv[1], input[1]) \
   << pq.CNOT(qv[0], qv[2])
prog = pq.QProg() << qc
pauli_dict = {'Z0 X1': 10, 'Y2': -0.543}
exp2 = expval(qvm, prog, pauli_dict, qv)
print('expval:', exp2)   # => 0.9983341664682827

''' DensityMatrixFromQstate & VN_Entropy & Mutal_Info '''
qstate = [
   0.9022+0j, -0.0667j,
   0.1829+0j, -0.3293j,
   0.0370+0j, -0.0667j,
   0.1829+0j, -0.0135j,
]
print(qstate)
print('DensityMatrix:', DensityMatrixFromQstate(qstate, [0, 2]))  # => 2d np.array
print('VN_Entropy:', VN_Entropy(qstate, [0, 2]))                  # float
print('Mutal_Info:', Mutal_Info(qstate, [0], [2], 2))             # float
print('Purity:', Purity(qstate, [1]))                             # float
