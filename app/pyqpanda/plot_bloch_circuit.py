#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/20 

py_eval = eval
from traceback import print_exc
from pyqpanda import *
from pyqpanda.Visualization import plot_bloch_circuit as show

qvm = CPUQVM()
qvm.init_qvm()
q = qvm.qAlloc()

print('Examples:')
print('  H(q)')
print('  H(q) << RX(q, 0.8)')
print('---------------------')

try:
  while True:
    s = input('Input your circuit: ')
    if s == 'q': break

    try:
      py_eval(f'show(QCircuit() << {s})')
    except:
      print_exc()
except KeyboardInterrupt:
  pass
