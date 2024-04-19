#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/20

# API list of pyqpanda-alg v1.0.0
# https://pyqpanda-algorithm-tutorial.readthedocs.io/en/latest/

from unittest import TestCase

from pyqpanda_alg.QLuoShu import ConModAdd, ConModaddmul, ConModExp, ConModMul, lshift, q_elliptic_padd, q_elliptic_pdou, QFTConAdd, VarModAdd, VarModDou, VarModInv, VarModMul, VarModNeg, VarModSqr
from pyqpanda_alg.QSolver import qsolver
from pyqpanda_alg.VQE import vqe
from pyqpanda_alg.QAOA import qaoa, dstate, spsa, default_circuits
from pyqpanda_alg.QFinance import comparator, grover, QAE, QUBO
from pyqpanda_alg.QPCA import QPCA
from pyqpanda_alg.QKmeans import QuantumKmeans
from pyqpanda_alg.QSVM import quantum_kernel_svm
from pyqpanda_alg.QARM import qarm


class TestAlgorithm(TestCase):

  def test_QLuoShu(self):   # quantum arithmetic
    pass

  def test_QSolver(self):   # mixed-HHL solver
    pass

  def test_VQE(self):
    pass

  def test_QAOA(self):
    pass

  def test_QFinance(self):  # amplitude estimation and Grover
    pass

  def test_QPCA(self):
    pass

  def test_QKmeans(self):
    pass

  def test_QSVM(self):
    pass

  def test_QARM(self):      # association rule mining
    pass
