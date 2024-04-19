#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/04 

import unittest as ut

import numpy as np
from pyqpanda import *


class TestCase(ut.TestCase):

  def test_QProg_API(self):
    # (12) API
    assert QProg
    # (2) add elem
    QProg.insert
    QProg.__lshift__
    # (4) iter
    QProg.begin
    QProg.end
    QProg.head
    QProg.last
    # (6) info
    QProg.get_max_qubit_addr
    QProg.get_qgate_num
    QProg.get_used_qubits
    QProg.get_used_cbits
    QProg.is_empty
    QProg.is_measure_last_pos   # this is broken

    ''' Test '''
    qvm = CPUQVM()
    qvm.init_qvm()
    q = qvm.qAlloc_many(2)
    c = qvm.cAlloc_many(2)
    qv = QVec()
    cv = []

    p = QProg()
    assert p.is_empty()
    #assert not p.is_measure_last_pos()
    p << H(q[0])
    p << X(q[1])
    assert not p.is_empty()
    #assert not p.is_measure_last_pos()
    p << measure_all(q, c)
    assert p.is_measure_last_pos()
    print(p)

    print(p.get_max_qubit_addr())
    print(p.get_qgate_num())
    print(p.get_used_qubits(qv))
    print(len(p.get_used_qubits([*q, *q])))   # wierd behaviour
    print(p.get_used_cbits(cv))
    #assert cv
    #assert qv[0] == q[0] and qv[1] == q[1]
    #assert cv[0] == c[0] and cv[1] == c[1]

    it = p.begin()
    assert it
    while it != p.end():
      print('>')
      it = it.get_next()
    while it != p.begin():
      print('<')
      it = it.get_pre()

    it = p.head()
    assert it
    while it != p.last():
      print('>')
      it = it.get_next()
    while it != p.head():
      print('<')
      it = it.get_pre()

  def test_QProg(self):
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(4)
    cbits = qvm.cAlloc_many(4)

    # 构建量子程序
    prog = QProg() \
          << H(qubits[0]) \
          << X(qubits[1]) \
          << iSWAP(qubits[0], qubits[1]) \
          << CNOT(qubits[1], qubits[2]) \
          << H(qubits[3])

    print(prog)
    #draw_qprog(prog, 'pic', filename='test_cir_draw.png')

    clock_cycle = get_qprog_clock_cycle(prog, qvm)
    print('clock_cycle:', clock_cycle)

    result_mat = get_matrix(prog)
    print_matrix(result_mat)

    # 量子程序运行1000次，并返回测量结果
    prog << measure_all(qubits, cbits)
    result = qvm.run_with_configuration(prog, cbits, 1000)
    # 打印量子态在量子程序多次运行结果中出现的次数
    print(result)

  def test_QIf_API(self):
    # (3) API
    assert QIfProg
    QIfProg.get_classical_condition
    QIfProg.get_true_branch
    QIfProg.get_false_branch

    ''' Test '''
    qvm = CPUQVM()
    qvm.init_qvm()

    q = qvm.qAlloc_many(2)
    c = qvm.cAlloc_many(2)
    c[0].set_val(114)
    c[1].set_val(514)

    qif = QIfProg(
      c[0] > 233 or c[1] < 1919,      # 114 > 233 or 514 < 1919 => True
      QProg() << H(q[0]),             # <= sir, this way
      QProg() << X(q[0]))

    print('qif:')
    print(QProg() << qif)
    print('cond:')
    print(qif.get_classical_condition())
    print('branch true:')
    print(qif.get_true_branch())
    print('branch false:')
    print(qif.get_false_branch())

    qif2 = QIfProg(
      c[0] + 1 > c[1],               # 114 + 1 > 514 => False
      qif.get_false_branch(),
      QProg() << qif << CNOT(q[0], q[1]))     # <= this way

    # cannot print if has undecided QIf
    #print(QProg() << qif2)
    #print(qif2.get_classical_condition())
    #print(qif2.get_true_branch())
    #print(qif2.get_false_branch())

    prog = QProg() << qif2
    # cannot print if has undecided QIf
    #print(prog)
    qvm.directly_run(prog)
    print(qvm.get_qstate())

    prog_eqv = QProg() << H(q[0]) << CNOT(q[0], q[1])
    print('prog_eqv:')
    print(prog_eqv)
    qvm.directly_run(prog_eqv)
    print(qvm.get_qstate())

  def test_QIf(self):
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(3)
    cbits = qvm.cAlloc_many(3)
    cbits[0].set_val(0)
    cbits[1].set_val(3)

    qif = QIfProg(
      cbits[0] > cbits[1] and cbits[0] < cbits[1],
      QProg() << H(qubits[0]) << H(qubits[1]) << H(qubits[2]), 
      QProg() << H(qubits[0]) << CNOT(qubits[0], qubits[1]) << CNOT(qubits[1], qubits[2]))

    prog = QProg() << qif
    print(prog)

    result = qvm.prob_run_dict(prog, qubits, -1)
    print(result)

  def test_QIf2(self):
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(3)
    cbits = qvm.cAlloc_many(2)
    cbits[0].set_val(0)
    cbits[1].set_val(3)

    branch_true  = QProg() << H(qubits)
    branch_false = QProg() << H(qubits[0]) << CNOT(qubits[0], qubits[1]) << CNOT(qubits[1], qubits[2])
    prog_if = QIfProg(cbits[0] > cbits[1], branch_true, branch_false)

    result = qvm.prob_run_tuple_list(prog_if, qubits)
    print(result)

  def test_QWhile_API(self):
    # (2) API
    assert QWhileProg
    QWhileProg.get_classical_condition
    QWhileProg.get_true_branch

    ''' Test '''
    qvm = CPUQVM()
    qvm.init_qvm()

    q = qvm.qAlloc_many(1)
    c = qvm.cAlloc_many(2)
    c[0].set_val(0)   # coin tossing result
    c[1].set_val(0)   # counter

    # NOTE: a very unfair coin: P(0) = 95%
    coin = QProg() \
      << RY(q[0], np.pi/7) \
      << assign(c[1], c[1] + 1) \
      << measure_all(q, [c[0]])         # measure q[0] => c[0]
    # toss the coin until it gets an `1``
    prog = QProg() << QWhileProg(c[0] == 0, coin)

    print('prog:')
    print(prog)
    qvm.directly_run(prog)
    print(qvm.get_qstate())
    print('coin:',    c[0].get_val())   # expect value == 1
    print('counter:', c[1].get_val())   # FIXME: expect value > 0

  def test_QWhile(self):
    init(QMachineType.CPU)
    qubits = qAlloc_many(3)
    cbits = cAlloc_many(3)
    cbits[0].set_val(0)
    cbits[1].set_val(1)

    prog_while = QProg() \
      << H(qubits[0]) \
      << H(qubits[1]) \
      << H(qubits[2]) \
      << assign(cbits[0], cbits[0] + 1) \
      << Measure(qubits[1], cbits[1])

    qwhile = create_while_prog(cbits[1], prog_while)

    prog = QProg() << qwhile
    result = directly_run(prog)
    print(result)
    print(cbits[0].get_val())
    print(cbits[1].get_val())
    finalize()

  def test_QWhile2(self):
    qvm = CPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(2)
    cbits = qvm.cAlloc_many(2)
    cbits[0].set_val(0)   # counter
    cbits[1].set_val(1)   # condition

    while_body = QProg() << H(qubits) << assign(cbits[0], cbits[0]+1) << Measure(qubits[1], cbits[1])
    prog_while = QWhileProg(cbits[1], while_body)

    result = qvm.directly_run(prog_while)
    print(result)
    print(cbits[0].get_val())
