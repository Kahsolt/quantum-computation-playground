#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/20

import unittest as ut
from traceback import print_exc

import numpy as np
from pyqpanda import *
from pyqpanda.Visualization.circuit_draw import *

# applications
from pyqpanda import QAdd, QAdder, QAdderIgnoreCarry, QSub, QMul, QMultiplier, QDiv, QDivWithAccuracy, QDivider, QDividerWithAccuracy, QComplement
from pyqpanda import isCarry, constModAdd, constModMul, constModExp, MAJ, MAJ2, UMA
from pyqpanda import QPE, QFT, Shor_factorization, iterative_amplitude_estimation
from pyqpanda import QITE, UpdateMode
from pyqpanda import Grover, Grover_search
from pyqpanda import HHLAlg, build_HHL_circuit, expand_linear_equations, HHL_solve_linear_equations
from pyqpanda import quantum_walk_alg, quantum_walk_search
from pyqpanda import em_method, QuantumStateTomography


class TestCase(ut.TestCase):

  def test_arithmetic(self):
    init()
    a = qAlloc_many(1)
    b = qAlloc_many(1)
    c = qAlloc_many(3)
    d = qAlloc_many(3)
    e = cAlloc()

    print('[QAdd]') ; print(QAdd(a, b, c))
    print('[QSub]') ; print(QSub(a, b, c))
    print('[QMul]') ; print(QMul(a, b, c, d))
    print('[QDiv]') ; print(QDiv(a, b, c, d, e))
    finalize()

  def test_bell_state(self):
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(2)
    cbits  = qvm.cAlloc_many(2)

    prog = QProg() \
      << H(qubits) \
      << CNOT(qubits[0], qubits[1]) \
      << RY(qubits[1], np.pi/8) \
      << CNOT(qubits[0], qubits[1]) \
      << RY(qubits[1], -np.pi/8)
    print(prog)
    #draw_qprog(prog, 'pic', filename='test_cir_draw.png')

    clock_cycle = get_qprog_clock_cycle(prog, qvm)
    print('clock_cycle:', clock_cycle)

    result_mat = get_matrix(prog)
    print_matrix(result_mat)

    prog << measure_all(qubits, cbits)
    result = qvm.run_with_configuration(prog, cbits, 1000)
    print(result)
