#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(3)
cbits = qvm.cAlloc_many(2)
cbits[0].set_val(0)
cbits[1].set_val(3)

# 构建QIf正确分支以及错误分支
branch_true  = QProg() << H(qubits)
branch_false = QProg() << H(qubits[0]) << CNOT(qubits[0], qubits[1]) << CNOT(qubits[1], qubits[2])

# 构建QIf
prog_if = QIfProg(cbits[0] > cbits[1], branch_true, branch_false)

# 概率测量，并返回目标量子比特的概率测量结果，下标为十进制
result = qvm.prob_run_tuple_list(prog_if, qubits)

# 打印概率测量结果
print(result)
