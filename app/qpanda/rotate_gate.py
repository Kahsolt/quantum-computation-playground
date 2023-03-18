#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

# RX 门可以改变 |0>/|1> 振幅比例，实部/虚部 数值比例
# RY 门可以改变 |0>/|1> 振幅比例，不改变虚部
# RZ 门不改变 |0>/|1> 振幅比例，实部/虚部 数值比例

#               RX                     |                  RY                  |                        RZ
#                                      |                                      |
# (6.123234e-17, 0)             (0, 1) | (6.123234e-17, 0)             (1, 0) |        (6.123234e-17, 1)                     (0, 0)
#            (0, 1)  (6.123234e-17, 0) |           (-1, 0)  (6.123234e-17, 0) |                   (0, 0)         (6.123234e-17, -1)
#                                      |                                      |
# (0.38268343, 0)  (0, 0.92387953)     |  (0.38268343, 0)  (0.92387953, 0)    | (0.38268343, 0.92387953)                     (0, 0)
# (0, 0.92387953)  (0.38268343, 0)     | (-0.92387953, 0)  (0.38268343, 0)    |                   (0, 0)  (0.38268343, -0.92387953)
#                                      |                                      |
# (0.70710678, 0)  (0, 0.70710678)     |  (0.70710678, 0)  (0.70710678, 0)    | (0.70710678, 0.70710678)                     (0, 0)
# (0, 0.70710678)  (0.70710678, 0)     | (-0.70710678, 0)  (0.70710678, 0)    |                   (0, 0)  (0.70710678, -0.70710678)
#                                      |                                      |
# (0.92387953, 0)  (0, 0.38268343)     |  (0.92387953, 0)  (0.38268343, 0)    | (0.92387953, 0.38268343)                     (0, 0)
# (0, 0.38268343)  (0.92387953, 0)     | (-0.38268343, 0)  (0.92387953, 0)    |                   (0, 0)  (0.92387953, -0.38268343)
#                                      |                                      |
#          (1, 0)          (0, -0)     |          (1, 0)           (-0, 0)    |                  (1, -0)                     (0, 0)
#         (0, -0)           (1, 0)     |          (0, 0)            (1, 0)    |                   (0, 0)                     (1, 0)
#                                      |                                      |
#  (0.92387953, 0)  (0, -0.38268343)   | (0.92387953, 0)  (-0.38268343, 0)    | (0.92387953, -0.38268343)                    (0, 0)
# (0, -0.38268343)   (0.92387953, 0)   | (0.38268343, 0)   (0.92387953, 0)    |                    (0, 0)  (0.92387953, 0.38268343)
#                                      |                                      |
#  (0.70710678, 0)  (0, -0.70710678)   | (0.70710678, 0)  (-0.70710678, 0)    | (0.70710678, -0.70710678)                    (0, 0)
# (0, -0.70710678)   (0.70710678, 0)   | (0.70710678, 0)   (0.70710678, 0)    |                    (0, 0)  (0.70710678, 0.70710678)
#                                      |                                      |
#  (0.38268343, 0)  (0, -0.92387953)   | (0.38268343, 0)  (-0.92387953, 0)    | (0.38268343, -0.92387953)                    (0, 0)
# (0, -0.92387953)   (0.38268343, 0)   | (0.92387953, 0)   (0.38268343, 0)    |                    (0, 0)  (0.38268343, 0.92387953)
#                                      |                                      |
# (6.123234e-17, 0)            (0, -1) | (6.123234e-17, 0)            (-1, 0) |        (6.123234e-17, -1)                    (0, 0)
#           (0, -1)  (6.123234e-17, 0) |            (1, 0)  (6.123234e-17, 0) |                    (0, 0)         (6.123234e-17, 1)


import numpy as np
from pyqpanda import *

qvm = CPUQVM()
qvm.init_qvm()
qubits = qvm.qAlloc_many(1)
cbits  = qvm.cAlloc_many(1)

for gate in [RX, RY, RZ]:
  for phi in np.linspace(-np.pi, np.pi, 9):
    print(gate.__name__, phi)

    prog = QProg() << gate(qubits[0], phi)
    print_matrix(get_matrix(prog))

    prog << measure_all(qubits, cbits)
    result = qvm.run_with_configuration(prog, cbits, 1000)
    print(result)
    print()
