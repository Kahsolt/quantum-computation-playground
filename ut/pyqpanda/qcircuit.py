#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

import random
import unittest as ut

import numpy as np
from pyqpanda import *

# gate
from pyqpanda import QGate, I, H, X, Y, Z, RX, RY, RZ, P, S, T, U1, U2, U3, U4, X1, Y1, Z1, RXX, RYY, RZZ, RZX, CNOT, CP, CR, CU, CZ, SWAP, iSWAP, SqiSWAP, Toffoli, GateType
from pyqpanda import QDouble, QOracle, MS, BARRIER
from pyqpanda import matrix_decompose, matrix_decompose_paulis, ldd_decompose, DecompositionMode, decompose_multiple_control_qgate, transform_to_base_qgate, transfrom_pauli_operator_to_matrix, virtual_z_transform
from pyqpanda import QReset, Reset, QMeasure, Measure, measure_all, pmeasure, PMeasure, pmeasure_no_index, PMeasure_no_index

# circuit & prog
from pyqpanda import QCircuit, create_empty_circuit, CreateEmptyCircuit
from pyqpanda import QProg, QIfProg, QWhileProg, create_empty_qprog, CreateEmptyQProg, create_if_prog, CreateIfProg, create_while_prog, CreateWhileProg, ClassicalProg, NodeIter, NodeInfo, NodeType
from pyqpanda import Fusion, QCircuitOPtimizerMode, SingleGateTransferType, DoubleGateTransferType
from pyqpanda import get_clock_cycle, get_qprog_clock_cycle, get_qgate_num, count_gate, count_qgate_num, count_prog_info
from pyqpanda import cast_qprog_qcircuit, cast_qprog_qgate, cast_qprog_qmeasure

from pyqpanda import (
  # QProg <-> OriginIR
  transform_qprog_to_originir, convert_qprog_to_originir, to_originir,
  transform_originir_to_qprog, convert_originir_to_qprog, convert_originir_str_to_qprog, originir_to_qprog,
  # QProg <-> Quil
  transform_qprog_to_quil, convert_qprog_to_quil, to_Quil,
  # QProg <-> QASAM
  convert_qprog_to_qasm,
  convert_qasm_to_qprog, convert_qasm_string_to_qprog,
  # QProg <-> Binary
  get_bin_data, get_bin_str,
  transform_qprog_to_binary, convert_qprog_to_binary, 
  transform_binary_data_to_qprog, convert_binary_data_to_qprog, bin_to_prog,
)


CU_orig = CU
CU = lambda q1, q2, a, b, c, d: CU_orig(a, b, c, d, q1, q2)

S_P0_GATE = [I, H, X, Y, Z, S, T, X1, Y1, Z1]
S_P1_GATE = [RX, RY, RZ, P, U1]
S_P2_GATE = [U2]
S_P3_GATE = [U3]
S_P4_GATE = [U4]
S_GATE = S_P0_GATE + S_P1_GATE + S_P2_GATE + S_P3_GATE + S_P4_GATE
D_P0_GATE = [CNOT, CZ, SWAP, iSWAP, SqiSWAP]
D_P1_GATE = [RXX, RYY, RZZ, RZX, CP, CR]
D_P4_GATE = [CU]
D_GATE = D_P0_GATE + D_P1_GATE + D_P4_GATE 
T_P0_GATE = [Toffoli]
T_GATE = T_P0_GATE
GATE = S_GATE + D_GATE + T_GATE


class TestCase(ut.TestCase):
  
  def test_QGate_API(self):
    # (11) API
    assert QGate
    # (2) info
    QGate.gate_type
    QGate.gate_matrix
    QGate.get_qubits
    # (5) control
    QGate.control
    QGate.set_control
    QGate.get_control_qubits
    QGate.get_control_qubit_num
    QGate.get_target_qubit_num
    # (3) dagger
    QGate.dagger
    QGate.set_dagger
    QGate.is_dagger

    ''' Test '''
    qvm = CPUQVM()
    qvm.init_qvm()

    q = qvm.qAlloc_many(2)
    qv = QVec()   # buffer for readout

    g = H(q[0])
    print(g.gate_type())
    print(g.gate_type() == HADAMARD_GATE)
    print(g.gate_matrix())
    print(g.get_qubits(qv))

    print(g.control(q[1]))
    print(g.set_control(q[1]))
    print(g.get_control_qubits(qv))
    print(g.get_control_qubit_num())
    print(g.get_target_qubit_num())

    print(g.dagger())
    print(g.set_dagger(False))
    print(g.is_dagger())
    print(g.set_dagger(True))
    print(g.is_dagger())

    prog = QProg() << g
    print(qvm.prob_run_list(prog, q))
    qvm.directly_run(prog)
    print(qvm.get_qstate())

  def test_gate(self):
    ngate = 100
    for nq in range(1, 72+1):
      print('nq:', nq)
      vqm = CPUQVM()
      vqm.set_configure(max_qubit=72, max_cbit=72)
      vqm.init_qvm()
      qv = vqm.qAlloc_many(nq)

      cnt, s_cnt, d_cnt = 0, 0, 0
      qcir = QCircuit()
      for _ in range(ngate):
        gate = random.choice(GATE)

        if gate in S_GATE:
          cnt += 1
          s_cnt += 1
          idx = random.choice(range(nq))
          qb = qv[idx]
          if gate in S_P0_GATE:
            qcir << gate(qb)
          if gate in S_P1_GATE:
            qcir << gate(qb, *np.random.uniform(size=1).tolist())
          if gate in S_P2_GATE:
            qcir << gate(qb, *np.random.uniform(size=2).tolist())
          if gate in S_P3_GATE:
            qcir << gate(qb, *np.random.uniform(size=3).tolist())
          if gate in S_P4_GATE:
            qcir << gate(qb, *np.random.uniform(size=4).tolist())

        if nq < 2: continue        

        if gate in D_GATE:
          cnt += 1
          d_cnt += 1
          idx = random.sample(range(nq), 2)
          qb = [qv[i] for i in idx]

          if gate in D_P0_GATE:
            qcir << gate(*qb)
          if gate in D_P1_GATE:
            qcir << gate(*qb, *np.random.uniform(size=1).tolist())
          if gate in D_P4_GATE:
            qcir << gate(*qb, *np.random.uniform(size=4).tolist())

        if nq < 3: continue        

        if gate in T_GATE:
          cnt += 1
          idx = random.sample(range(nq), 3)
          qb = [qv[i] for i in idx]

          if gate in T_P0_GATE:
            qcir << gate(*qb)

      qprog = QProg() << qcir
      #print(get_clock_cycle(qprog))      # => Windows fatal exception: access violation
      #print(get_qprog_clock_cycle(qprog, vqm))    # => RuntimeError: Bad nodeType -> 9 run error
      self.assertLessEqual(get_qgate_num(qprog), ngate)
      self.assertLessEqual(count_gate(qprog), ngate)
      self.assertLessEqual(count_qgate_num(qprog), ngate)
      self.assertEqual(count_qgate_num(qprog), get_qgate_num(qprog))
      self.assertEqual(count_qgate_num(qprog), count_gate(qprog))
      info = count_prog_info(qprog)
      self.assertEqual(info.gate_num, cnt)
      #self.assertEqual(info.single_gate_num, s_cnt)    # <= error
      self.assertEqual(info.double_gate_num, d_cnt)

  def test_rot_gate(self):
    '''
                  RX                     |                  RY                  |                        RZ
                                         |                                      |
    (6.123234e-17, 0)             (0, 1) | (6.123234e-17, 0)             (1, 0) |        (6.123234e-17, 1)                     (0, 0)
               (0, 1)  (6.123234e-17, 0) |           (-1, 0)  (6.123234e-17, 0) |                   (0, 0)         (6.123234e-17, -1)
                                         |                                      |
    (0.38268343, 0)  (0, 0.92387953)     |  (0.38268343, 0)  (0.92387953, 0)    | (0.38268343, 0.92387953)                     (0, 0)
    (0, 0.92387953)  (0.38268343, 0)     | (-0.92387953, 0)  (0.38268343, 0)    |                   (0, 0)  (0.38268343, -0.92387953)
                                         |                                      |
    (0.70710678, 0)  (0, 0.70710678)     |  (0.70710678, 0)  (0.70710678, 0)    | (0.70710678, 0.70710678)                     (0, 0)
    (0, 0.70710678)  (0.70710678, 0)     | (-0.70710678, 0)  (0.70710678, 0)    |                   (0, 0)  (0.70710678, -0.70710678)
                                         |                                      |
    (0.92387953, 0)  (0, 0.38268343)     |  (0.92387953, 0)  (0.38268343, 0)    | (0.92387953, 0.38268343)                     (0, 0)
    (0, 0.38268343)  (0.92387953, 0)     | (-0.38268343, 0)  (0.92387953, 0)    |                   (0, 0)  (0.92387953, -0.38268343)
                                         |                                      |
             (1, 0)          (0, -0)     |          (1, 0)           (-0, 0)    |                  (1, -0)                     (0, 0)
            (0, -0)           (1, 0)     |          (0, 0)            (1, 0)    |                   (0, 0)                     (1, 0)
                                         |                                      |
     (0.92387953, 0)  (0, -0.38268343)   | (0.92387953, 0)  (-0.38268343, 0)    | (0.92387953, -0.38268343)                    (0, 0)
    (0, -0.38268343)   (0.92387953, 0)   | (0.38268343, 0)   (0.92387953, 0)    |                    (0, 0)  (0.92387953, 0.38268343)
                                         |                                      |
     (0.70710678, 0)  (0, -0.70710678)   | (0.70710678, 0)  (-0.70710678, 0)    | (0.70710678, -0.70710678)                    (0, 0)
    (0, -0.70710678)   (0.70710678, 0)   | (0.70710678, 0)   (0.70710678, 0)    |                    (0, 0)  (0.70710678, 0.70710678)
                                         |                                      |
     (0.38268343, 0)  (0, -0.92387953)   | (0.38268343, 0)  (-0.92387953, 0)    | (0.38268343, -0.92387953)                    (0, 0)
    (0, -0.92387953)   (0.38268343, 0)   | (0.92387953, 0)   (0.38268343, 0)    |                    (0, 0)  (0.38268343, 0.92387953)
                                         |                                      |
    (6.123234e-17, 0)            (0, -1) | (6.123234e-17, 0)            (-1, 0) |        (6.123234e-17, -1)                    (0, 0)
              (0, -1)  (6.123234e-17, 0) |            (1, 0)  (6.123234e-17, 0) |                    (0, 0)         (6.123234e-17, 1)
    '''

    # RX 门可以改变 |0>/|1> 振幅比例，实部/虚部 数值比例
    # RY 门可以改变 |0>/|1> 振幅比例，不改变虚部
    # RZ 门不改变 |0>/|1> 振幅比例，实部/虚部 数值比例

    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(1)
    cbits  = qvm.cAlloc_many(1)

    for gate in [RX, RY, RZ]:
      for phi in np.linspace(-np.pi, np.pi, 9):
        print(gate.__name__, phi)

        prog = QProg() << gate(qubits[0], phi)
        print_matrix(get_matrix(prog))

        prog << measure_all(qubits, cbits)
        result = qvm.run_with_configuration(prog, cbits, 1000)
        print(result)
        print()

  def test_decomp(self):
    nq = 4
    vqm = CPUQVM()
    vqm.init_qvm()
    qv = vqm.qAlloc_many(nq)

    qcir = QCircuit() \
      << I(qv[0]).control(qv[3]) \
      << H(qv[0]).control(qv[3]) \
      << X(qv[1]).control(qv[2]) \
      << Y(qv[1]).control(qv[2]) \
      << Z(qv[1]).control(qv[2]) \
      << S(qv[0]).control(qv[3]) \
      << T(qv[0]).control(qv[3]) \
      << P(qv[1], 0.1).control(qv[2]) \
      << X1(qv[1]).control(qv[2]) \
      << Y1(qv[1]).control(qv[2]) \
      << Z1(qv[1]).control(qv[2]) \
      << U1(qv[1], 0.1).control(qv[2]) \
      << U2(qv[1], 0.2, 0.3).control(qv[2]) \
      << U3(qv[1], 0.4, 0.5, 0.6).control(qv[2]) \
      << U4(qv[1], 0.7, 0.8, 0.9, 0.1).control(qv[2]) \
      << RX(qv[3], 1.2).control(qv[2]) \
      << RZ(qv[3], 1.2).control(qv[2]) \
      << RY(qv[3], 1.2).control(qv[2]) \
      << CR(qv[3], qv[2], 1.2) \
      << CP(qv[3], qv[2], 1.2) \
      << CNOT(qv[2], qv[0]).dagger() \
      << CZ(qv[2], qv[0]).dagger() \
      << SWAP(qv[1], qv[3]) \
      << SqiSWAP(qv[1], qv[3])

    qprog = QProg() << qcir

    source_matrix = get_unitary(qprog)
    print("source_matrix: ")
    print(source_matrix)
    out_cir = matrix_decompose(qv, np.array(source_matrix).reshape(2**nq, 2**nq))
    circuit_matrix = get_unitary(out_cir)
    print("the decomposed matrix: ")
    print(circuit_matrix)

    ldd_decompose(qprog)
    decompose_multiple_control_qgate(qprog, vqm)
    transform_to_base_qgate(qprog, vqm)
    virtual_z_transform(qprog, vqm)

  def test_transcription(self):
    nq = 4
    vqm = CPUQVM()
    vqm.init_qvm()
    qv = vqm.qAlloc_many(nq)

    qcir = QCircuit() \
      << I(qv[0]).control(qv[3]) \
      << H(qv[0]).control(qv[3]) \
      << X(qv[1]).control(qv[2]) \
      << Y(qv[1]).control(qv[2]) \
      << Z(qv[1]).control(qv[2]) \
      << S(qv[0]).control(qv[3]) \
      << T(qv[0]).control(qv[3]) \
      << X1(qv[1]).control(qv[2]) \
      << Y1(qv[1]).control(qv[2]) \
      << Z1(qv[1]).control(qv[2]) \
      << U1(qv[1], 0.1).control(qv[2]) \
      << U2(qv[1], 0.2, 0.3).control(qv[2]) \
      << U3(qv[1], 0.4, 0.5, 0.6).control(qv[2]) \
      << U4(qv[1], 0.7, 0.8, 0.9, 0.1).control(qv[2]) \
      << RX(qv[3], 1.2).control(qv[2]) \
      << RZ(qv[3], 1.2).control(qv[2]) \
      << RY(qv[3], 1.2).control(qv[2]) \
      << CR(qv[3], qv[2], 1.2) \
      << CNOT(qv[2], qv[0]).dagger() \
      << CZ(qv[2], qv[0]).dagger() \
      << SWAP(qv[1], qv[3]) \
      << SqiSWAP(qv[1], qv[3])
      #<< CP(qv[3], qv[2], 1.2) \
      #<< P(qv[1], 0.1).control(qv[2]) \

    qprog = QProg() << qcir

    ir1 = transform_qprog_to_originir(qprog, vqm)
    ir2 = convert_qprog_to_originir(qprog, vqm)
    ir3 = to_originir(qprog, vqm)
    self.assertEqual(ir1, ir2)
    self.assertEqual(ir1, ir3)
    convert_originir_str_to_qprog(ir1, vqm)
    convert_originir_str_to_qprog(ir2, vqm)
    convert_originir_str_to_qprog(ir3, vqm)

    # QProg <-> Quil
    quil1 = transform_qprog_to_quil(qprog, vqm)
    quil2 = convert_qprog_to_quil(qprog, vqm)
    quil3 = to_Quil(qprog, vqm)
    self.assertEqual(quil1, quil2)
    self.assertEqual(quil1, quil3)

    # QProg <-> QASAM
    qsam = convert_qprog_to_qasm(qprog, vqm)
    convert_qasm_string_to_qprog(qsam, vqm)

    # QProg <-> Binary
    #get_bin_data(qprog)      # -> Windows fatal exception: access violation
    get_bin_str(qprog, vqm)
    bin1 = transform_qprog_to_binary(qprog, vqm)
    bin2 = convert_qprog_to_binary(qprog, vqm)
    self.assertEqual(bin1, bin2)
    transform_binary_data_to_qprog(vqm, bin1)
    convert_binary_data_to_qprog(vqm, bin1)

  def test_transcription2(self):
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(3)
    cbits  = qvm.cAlloc_many(3)

    qif = QIfProg(
      cbits[0] > cbits[1] and cbits[2] < cbits[1],
      QProg() << H(qubits[0]) << H(qubits[1]) << H(qubits[2]), 
      QProg() << H(qubits[0]) << CNOT(qubits[0], qubits[1]) << CNOT(qubits[1], qubits[2]))
    prog = QProg() << qif

    info = to_originir(prog, qvm)
    print(info)

    prog = QProg() \
        << H(qubits) \
        << assign(cbits[0], cbits[0] + 1) \
        << Measure(qubits[1], cbits[1])
    prog = QProg() << create_while_prog(cbits[0] < 10, prog)
    info = to_originir(prog, qvm)
    print(info)

  def test_QCircuit_API(self):
    # (12) API, gate-like QProg
    assert QCircuit
    # (2) add elem
    QCircuit.insert
    QCircuit.__lshift__
    # (4) iter
    QCircuit.begin
    QCircuit.end
    QCircuit.head
    QCircuit.last
    # (5) gate-like
    QCircuit.control
    QCircuit.set_control
    QCircuit.dagger
    QCircuit.set_dagger
    QCircuit.is_empty

    ''' Test '''
    qvm = CPUQVM()
    qvm.init_qvm()
    q = qvm.qAlloc_many(3)

    cq = QCircuit()
    assert cq.is_empty()
    cq << RY(q[0], 1.2) \
      << RZ(q[1], -1.8) \
      << CNOT(q[1], q[0])
    assert not cq.is_empty()

    print('original:')
    print(cq)

    print('control:')
    print(cq.control(q[2]))   # Control(q[2], cq), q2 over all cq as a gate
    cq.set_control(q[2])
    print('set_control:')
    print(cq)

    print('dagger:')
    print(cq.dagger())
    cq.set_dagger(False)
    cq.set_dagger(True)
