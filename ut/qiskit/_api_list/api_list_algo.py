#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

from pprint import pprint as pp
import qiskit.algorithms as QA

names = [name for name in dir(QA) if not name.startswith('_')]
pp(names)
