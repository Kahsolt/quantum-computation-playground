#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

import qiskit_machine_learning as QML
import qiskit_machine_learning.algorithms as QAlgo
import qiskit_machine_learning.circuit as QCirq
import qiskit_machine_learning.datasets as QDS
import qiskit_machine_learning.neural_networks as QNN

qgan = QAlgo.QGAN
print(qgan)
qgan.discriminator
nn = QNN.TwoLayerQNN(2)
print(nn)
