#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/20 

from pyqpanda.Operator.pyQPandaOperator import *

H1 = PauliOperator({'X0':  1})
H2 = PauliOperator({'Y0': -1})
a = 2 + H1
b = 3 + H2
c = a * b
d = 4 * c
print(d)
print('='*40)

m1 = PauliOperator({'X2 z1 y3 x4': 5.0, "z0 X2 y5": 8})
m2 = PauliOperator({'X2 z0 y3 x4 y1': -5})
print(m1)
print(m2)
print(m1 + m2)
print(m1 - m2)
print(m1 * m2)
print('='*40)


exit()
# FIXME: what is a valid ferm-op ?
H1 = FermionOperator({'X0':  1})
H2 = FermionOperator({'Y0': -1})
a = 2 + H1
b = 3 + H2
c = a * b
d = 4 * c
print(d)
