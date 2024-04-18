#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/04 

from projectq import MainEngine
from projectq.ops import H, Measure

eng = MainEngine()
qubit = eng.allocate_qubit()

# apply a Hadamard gate
H | qubit
# measure the qubit
Measure | qubit

# flush all gates (and execute measurements)
eng.flush()
# output measurement result
print(f"Measured {int(qubit)}")
