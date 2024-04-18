#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(2)
cbits = qvm.cAlloc_many(2)
cbits[0].set_val(0)   # counter
cbits[1].set_val(1)   # condition

# 构建QWhile的循环分支
while_body = QProg() << H(qubits) << assign(cbits[0], cbits[0]+1) << Measure(qubits[1], cbits[1])

# 构建QWhile
prog_while = QWhileProg(cbits[1], while_body)

# 运行，并打印测量结果
result = qvm.directly_run(prog_while)
print(result)
print(cbits[0].get_val())

# NOTE: 好像结果不太对……