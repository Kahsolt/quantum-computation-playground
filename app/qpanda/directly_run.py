#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(4)
cbits  = qvm.cAlloc_many(4)

# 构建量子程序
prog = QProg() \
      << H(qubits[0]) \
      << X(qubits[1]) \
      << iSWAP(qubits[0], qubits[1]) \
      << CNOT(qubits[1], qubits[2]) \
      << H(qubits[3])

print(prog)

print(qvm.directly_run(prog))
print(qvm.quick_measure(prog.get_used_qubits([]), shots=1000))

print(qvm.run_with_configuration(prog << measure_all(qubits, cbits), shot=1000))
