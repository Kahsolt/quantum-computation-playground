#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/23 

import numpy as np
import pyqpanda as pq
from pyvqnet import tensor
from pyvqnet.qnn.ansatz import HardwareEfficientAnsatz
from pyvqnet.qnn.template import (
  RandomTemplate,
  BasicEmbeddingCircuit,
  AmplitudeEmbeddingCircuit,
  AngleEmbeddingCircuit,
  IQPEmbeddingCircuits,
  BasicEntanglerTemplate,
  StronglyEntanglingTemplate,
  RotCircuit,
  CRotCircuit,
  CSWAPcircuit,
)

# all tmpl has following method
assert RandomTemplate.compute_circuit
assert RandomTemplate.create_circuit
assert RandomTemplate.print_circuit
# except HardwareEfficientAnsatz
assert HardwareEfficientAnsatz.get_para_num
assert HardwareEfficientAnsatz.create_ansatz

n_qubits = 3
qvm = pq.CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(n_qubits)


''' data encoder (functions) '''
print('BasicEmbeddingCircuit: bool(b) -> |b>')
inputs = np.asarray([1, 0, 1])
qc = BasicEmbeddingCircuit(inputs, qubits)
print(qc)

print('AngleEmbeddingCircuit: float(α) -> RY(α)|0>')
inputs = np.array([1.2, 2.4, -3.6])
qc = AngleEmbeddingCircuit(inputs, qubits, 'Y')
print(qc)

print('AmplitudeEmbeddingCircuit: 2^n cbits -> n qubits')
inputs = np.random.uniform(low=-4, high=4, size=[2**n_qubits])
inputs /= inputs.sum()
qc = AmplitudeEmbeddingCircuit(inputs, qubits)
print(qc)

print('IQPEmbeddingCircuits: n cbits -> n qubits')
inputs = np.array([1.2, 2.4, -3.6])
qc = IQPEmbeddingCircuits(inputs, qubits, rep=1)
print(qc)

''' circuit snippets (functions) '''
print('RotCircuit: single-qubit RZ-RY-RZ rotation')
param = np.random.random(size=[3])
qc = RotCircuit(param, qubits[:1])
print(qc)

print('RotCircuit: single-qubit controlled single-qubit RZ-RY-RZ rotation')
param = np.random.random(size=[3])
qc = CRotCircuit(param, qubits[1:2], qubits[:1])
print(qc)

print('CSWAPcircuit: controlled swap')
assert len(qubits) == 3
qc = CSWAPcircuit(qubits)
print(qc)


''' ansatz (classes) '''
print('RandomTemplate: random circuit')
weights = np.random.random(size=[1, n_qubits*2])
tmpl = RandomTemplate(weights, num_qubits=n_qubits, seed=None)    # leave seed unset
tmpl.print_circuit(qubits)        # will yield different circuit at each call
qc = tmpl.create_circuit(qubits)  # will yield different circuit at each call

print('BasicEntanglerTemplate: RX/RY/RZ + CNOT cycle')
weights = np.random.random(size=[1, n_qubits])
tmpl = BasicEntanglerTemplate(weights=weights, num_qubits=n_qubits, rotation=pq.RZ)
tmpl.print_circuit(qubits)
qc = tmpl.create_circuit(qubits)

print('StronglyEntanglingTemplate: RX-RY-RZ + CNOT cycle')
weights = np.random.random(size=[1, n_qubits, 3])
tmpl = StronglyEntanglingTemplate(weights=weights, num_qubits=n_qubits)
tmpl.print_circuit(qubits)
qc = tmpl.create_circuit(qubits)

print('HardwareEfficientAnsatz: RX?RY?RZ? + CNOT/CZ linear/full')
tmpl = HardwareEfficientAnsatz(n_qubits, ["rx", "ry", "rz"], qubits, entangle_gate="cnot", entangle_rules="linear", depth=1)    # 'cnot'/'cz', 'linear'/'all'
qc = tmpl.create_ansatz(tensor.randu([tmpl.get_para_num()]))
print(qc)
