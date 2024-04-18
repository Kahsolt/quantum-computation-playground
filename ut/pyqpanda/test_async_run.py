#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/02 

import random
from time import sleep
from pyqpanda import *

pi = 3.1415926

qvm = CPUQVM()
qvm.init_qvm()
nq = 20
qv = qvm.qAlloc_many(nq)
cv = qvm.cAlloc_many(nq)

qc = QCircuit()
for _ in range(10000):
  q = qv[random.randrange(nq)]

  if random.random() < 0.85:
    r = random.random()
    if   r < 0.2: qc << H(q)
    elif r < 0.3: qc << X(q)
    elif r < 0.4: qc << Y(q)
    elif r < 0.5: qc << Z(q)
    elif r < 0.6: qc << RX(q, random.random() * pi)
    elif r < 0.7: qc << RY(q, random.random() * pi)
    elif r < 0.8: qc << RZ(q, random.random() * pi)

  if random.random() < 0.75:
    i = random.randrange(nq)
    j = random.randrange(nq)
    if i != j:
      r =  random.random()
      if r < 0.25:
        qc << CNOT(qv[i], qv[j])
      elif r < 0.5:
        qc << CR(qv[i], qv[j], random.random() * pi)
      elif r < 0.75:
        qc << SWAP(qv[i], qv[j])

prog = QProg() << qc << measure_all(qv, cv)

qvm.async_run(prog)
while not qvm.is_async_finished():
  print('>> processed', qvm.get_processed_qgate_num())
  sleep(0.1)

res = qvm.get_async_result()
print(res)
