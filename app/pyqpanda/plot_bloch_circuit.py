#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/20 

from traceback import print_exc

from pyqpanda import CPUQVM
from pyqpanda import H, X, Y, Z, RX, RY, RZ, S, T, U1   # all supported gates
from pyqpanda.Visualization import plot_bloch_circuit as show

print('Examples:')
print('  H(q)')
print('  H(q) << RX(q, 0.8)')
print('---------------------')


try:
  qvm = CPUQVM()
  qvm.init_qvm()
  q = qvm.qAlloc()

  while True:
    s = input('Input your circuit: ')
    if s == 'q': break

    try:
      show(eval(f'QCircuit() << {s}'))
    except:
      print_exc()
except KeyboardInterrupt:
  pass
