#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

import pyqpanda as pq
from traceback import print_exc

for N in range(1, 21):
  try:
    r = pq.Shor_factorization(N)
    print(f'N={N}', r)
  except:
    print_exc()
