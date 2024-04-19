#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/22

import random
import unittest as ut

# classical
from pyqpanda import CBit, ClassicalCondition, OriginCMem
from pyqpanda import add, sub, div, mul, assign, equal
from pyqpanda import CPUQVM
# qubit
from pyqpanda import Qubit, QVec, PhysicalQubit, OriginQubitPool
# operator
from pyqpanda.Operator.pyQPandaOperator import *


class TestCase(ut.TestCase):
  
  def test_CMem(self):
    cmem = OriginCMem()
    self.assertEqual(len(cmem.get_allocate_cbits()), 0)

    cb = cmem.Allocate_CBit()
    self.assertEqual(type(cb), CBit)
    self.assertEqual(len(cmem.get_allocate_cbits()), 1)
    cmem.Free_CBit(cb)
    self.assertEqual(len(cmem.get_allocate_cbits()), 0)

    cv = cmem.Allocate_CBit(10)   # FIXME: 入参无效，返回单个CBit
    self.assertEqual(type(cv), CBit)
    self.assertEqual(len(cmem.get_allocate_cbits()), 1)
    cmem.Free_CBit(cv)
    self.assertEqual(len(cmem.get_allocate_cbits()), 0)

    cb = cmem.cAlloc()
    self.assertEqual(len(cmem.get_allocate_cbits()), 1)
    self.assertEqual(type(cb), CBit)
    self.assertRaises(TypeError, lambda: cmem.cFree(cb))  # 签名错误
    cmem.Free_CBit(cb)
    # cmem.clearAll()   # => Windows fatal exception: access violation

    cv = cmem.cAlloc_many(10)
    self.assertEqual(len(cmem.get_allocate_cbits()), 10)
    self.assertEqual(type(cv), list)
    cmem.cFree_all(cv)

    cap = 10
    cmem.set_capacity(cap)
    self.assertEqual(cmem.get_capacity(), cap)  # => wrong number
    cap = 100
    cmem.set_capacity(cap)
    self.assertEqual(cmem.get_capacity(), cap)  # => wrong number

  def test_QPool(self):
    qpool = OriginQubitPool()
    self.assertEqual(len(qpool.get_allocate_qubits()), 0)

    qb = qpool.qAlloc()
    self.assertEqual(type(qb), Qubit)
    self.assertEqual(len(qpool.get_allocate_qubits()), 1)
    qpool.qFree(qb)
    self.assertEqual(len(qpool.get_allocate_qubits()), 0)

    qv = qpool.qAlloc_many(10)
    self.assertEqual(type(qv), list)
    self.assertEqual(len(qpool.get_allocate_qubits()), 10)
    qpool.qFree_all(qv)
    self.assertEqual(len(qpool.get_allocate_qubits()), 0)

    qpool.getIdleQubit()
    qpool.getMaxQubit()

    qpool.allocateQubitThroughPhyAddress(0)
    qpool.allocateQubitThroughVirAddress(0)

    cap = 10
    qpool.set_capacity(cap)
    self.assertEqual(qpool.get_capacity(), cap)
    cap = 100
    qpool.set_capacity(cap)
    self.assertEqual(qpool.get_capacity(), cap)

  def test_expression(self):
    vqm = CPUQVM()
    vqm.init_qvm()

    b1 = vqm.cAlloc()
    b2 = vqm.cAlloc()
    b_add = add(b1, b2)
    self.assertEqual(b_add.get_val(), b1.get_val() + b2.get_val())
    b_sub = sub(b1, b2)
    self.assertEqual(b_sub.get_val(), b1.get_val() - b2.get_val())
    b_mul = mul(b1, b2)
    self.assertEqual(b_mul.get_val(), b1.get_val() * b2.get_val())
    b_div = div(b1, b2)
    self.assertEqual(b_div.get_val(), b1.get_val() // b2.get_val())
    b_equ = equal(b1, b2)
    self.assertEqual(b_equ.get_val(), b1.get_val() == b2.get_val())
    b3 = assign(b1, b2)
    self.assertEqual(b3.get_val(), b1.get_val())

    b1.set_val(514)
    b2.set_val(114)
    b_add = add(b1, b2)
    self.assertEqual(b_add.get_val(), b1.get_val() + b2.get_val())
    b_sub = sub(b1, b2)
    self.assertEqual(b_sub.get_val(), b1.get_val() - b2.get_val())
    b_mul = mul(b1, b2)
    self.assertEqual(b_mul.get_val(), b1.get_val() * b2.get_val())
    b_div = div(b1, b2)
    self.assertEqual(b_div.get_val(), b1.get_val() // b2.get_val())
    b_equ = equal(b1, b2)
    self.assertEqual(b_equ.get_val(), b1.get_val() == b2.get_val())
    b3 = assign(b1, b2)
    self.assertEqual(b3.get_val(), b1.get_val())

    b1.set_val(1)
    b2.set_val(1)
    b3.set_val(0)
    # ~b1 & b2 | ~b3
    ex = b1.c_not().c_and(b2).c_or(b3.c_not())
    self.assertEqual(ex.get_val(), 1)

    vqm.cFree_all()
    vqm.finalize()

  def test_alloc(self):
    max_qubit = 72
    max_cbit = 1024

    vqm = CPUQVM()
    vqm.init_qvm()
    vqm.set_configure(max_qubit, max_cbit)

    alloc_qubits = []
    alloc_cbits = []
    free_qubit = max_qubit
    free_cbit = max_cbit
    for _ in range(10000):
      if random.random() < 0.5:
        if random.random() < 0.5:
          if free_qubit <= 0: continue
          cnt = random.randrange(1, free_qubit+1)
          alloc_qubits.extend(vqm.qAlloc_many(cnt))
          free_qubit -= cnt
        else:
          if free_cbit <= 0: continue
          cnt = random.randrange(1, free_cbit+1)
          alloc_cbits.extend(vqm.cAlloc_many(cnt))
          free_cbit -= cnt
      else:   # free
        if random.random() < 0.5:
          if len(alloc_qubits) < 0: continue
          cnt = min(1, int(len(alloc_qubits) * random.random()))
          qubits = random.sample(alloc_qubits, cnt)
          vqm.qFree_all(qubits)
          for qubit in qubits: alloc_qubits.remove(qubit)
          free_qubit += cnt
        else:
          if len(alloc_cbits) < 0: continue
          cnt = min(1, int(len(alloc_cbits) * random.random()))
          cbits = random.sample(alloc_cbits, cnt)
          vqm.cFree_all(cbits)
          for cbit in cbits: alloc_cbits.remove(cbit)
          free_cbit += cnt

      self.assertEqual(vqm.getAllocateQubitNum(), vqm.get_allocate_qubit_num())
      self.assertEqual(vqm.getAllocateCMem(), vqm.get_allocate_cmem_num())
      self.assertEqual(max_qubit - free_qubit, vqm.get_allocate_qubit_num())
      self.assertEqual(max_cbit - free_cbit, vqm.get_allocate_cmem_num())

  def test_QOperator(self):
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

    return
    # FIXME: what is a valid ferm-op ?
    H1 = FermionOperator({'X0':  1})
    H2 = FermionOperator({'Y0': -1})
    a = 2 + H1
    b = 3 + H2
    c = a * b
    d = 4 * c
    print(d)
