#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(4)
cbits = qvm.cAlloc_many(4)

# 构建量子程序
prog = QProg() \
      << H(qubits[0]) \
      << X(qubits[1]) \
      << iSWAP(qubits[0], qubits[1]) \
      << CNOT(qubits[1], qubits[2]) \
      << H(qubits[3])

print(prog)
#draw_qprog(prog, 'pic', filename='test_cir_draw.png')

clock_cycle = get_qprog_clock_cycle(prog, qvm)
print('clock_cycle:', clock_cycle)

result_mat = get_matrix(prog)
print_matrix(result_mat)

# 量子程序运行1000次，并返回测量结果
prog << measure_all(qubits, cbits)
result = qvm.run_with_configuration(prog, cbits, 1000)
# 打印量子态在量子程序多次运行结果中出现的次数
print(result)
