#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/20 

from pyqpanda import *

init()
a = qAlloc_many(1)
b = qAlloc_many(1)
c = qAlloc_many(3)
d = qAlloc_many(3)
e = cAlloc()

print('[QAdd]') ; print(QAdd(a, b, c))
print('[QSub]') ; print(QSub(a, b, c))
#print('[QMul]') ; print(QMul(a, b, c, d))
#print('[QDiv]') ; print(QDiv(a, b, c, d, e))

finalize()
