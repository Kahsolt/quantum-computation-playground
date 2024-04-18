#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/27 

# QAOA 是众所周知的量子经典混合算法。 对于n对象的MAX-CUT问题，需要n个量子位来对结果进行编码，其中测量结果（二进制串）表示问题的切割配置。
# Quantum Approximate Optimazation Algorithm (QAOA; 量子近似最適化アルゴリズム)
# https://dojo.qulacs.org/ja/latest/notebooks/5.3_quantum_approximate_optimazation_algorithm.html

from pyqpanda import *
from pyqpanda.Visualization.circuit_draw import *
import numpy as np

def oneCircuit(qlist, Hamiltonian, beta, gamma):
  vqc = VariationalQuantumCircuit()
  for i in range(len(Hamiltonian)):
    tmp_vec = []
    item = Hamiltonian[i]   # ({0: 'Z', 4: 'Z'}, 0.73)
    dict_p, coef = item     # {0: 'Z', 4: 'Z'}, 0.73
    for iter in dict_p:
      if 'Z' != dict_p[iter]: pass    # 应该没有不是 'Z' 的……
      tmp_vec.append(qlist[iter])     # 取得对应索引号的qubit
    if 2 != len(tmp_vec): pass        # 应该没有不是两个的……

    vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
    vqc.insert(VariationalQuantumGate_RZ  (tmp_vec[1], 2 * gamma * coef))
    vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))

  for j in qlist:
    vqc.insert(VariationalQuantumGate_RX(j, 2.0 * beta))
  return vqc


if __name__=="__main__":
  # dist 0 1 2 3 4 5 6
  #    0 0 2 2 2 1 1 1
  #    1   0 2 2 1 1 3
  #    2     0 2 3 1 1
  #    3       0 3 1 1
  #    4         0 2 2
  #    5           0 2
  #    6             0
  # Pauli算符 p("Z0 Z1", 2) 表示 2σz0⊗σz1，多个表示相加
  problem1 = {
    'Z0 Z3': 0.25,
    'Z1 Z3': 0.25,
    'Z1 Z4': 0.25,
    'Z2 Z3': 0.25,
    'Z2 Z4': 0.25,
  }
  problem2 = {
    'Z0 Z4': 0.73,
    'Z0 Z5': 0.33,
    'Z0 Z6': 0.5,
    'Z1 Z4': 0.69,
    'Z1 Z5': 0.36,
    'Z2 Z5': 0.88,
    'Z2 Z6': 0.58,
    'Z3 Z5': 0.67,
    'Z3 Z6': 0.43,
  }
  problem3 = {
    'Z0 Z4': 0.1,
    'Z0 Z5': 0.1,
    'Z0 Z6': 0.1,
    'Z1 Z4': 0.1,
    'Z1 Z5': 0.1,
    'Z2 Z5': 0.1,
    'Z2 Z6': 0.1,
    'Z3 Z5': 0.1,
    'Z3 Z6': 0.1,
  }
  Hp = PauliOperator(problem2)
  qubit_num = Hp.getMaxIndex() + 1
  print('qubit_num:', qubit_num)

  machine = init_quantum_machine(QMachineType.CPU)
  qlist = machine.qAlloc_many(qubit_num)

  step  = 4   # 论文中的整数 p
  beta  = var(np.ones((step, 1)).astype(np.float64), True)    # requires_grad=True
  gamma = var(np.ones((step, 1)).astype(np.float64), True)

  vqc = VariationalQuantumCircuit()
  for i in qlist:         # 制作初态 |s>
    vqc << VariationalQuantumGate_H(i)
  for i in range(step):
    vqc << oneCircuit(qlist, Hp.toHamiltonian(1), beta[i], gamma[i])

  loss = qop(vqc, Hp, machine, qlist)   # 哈密顿量的期望
  optimizer = MomentumOptimizer.minimize(loss, 0.02, 0.9)

  leaves = optimizer.get_variables()
  betas, gammas, losses = [], [], []
  for i in range(40):
    optimizer.run(leaves, 0)
    loss_value = optimizer.get_loss()
    losses.append(loss_value)
    print(f'i: {i}, loss: {loss_value}')

    betas .append(beta .get_value())
    gammas.append(gamma.get_value())
  
  betas  = np.concatenate(betas,  axis=-1)    # [4, T]
  gammas = np.concatenate(gammas, axis=-1)    # [4, T]

  # 验证结果
  prog = QProg() << vqc.feed()
  print(prog)
  directly_run(prog)
  result = quick_measure(qlist, 1000)
  print(result)

  # 画图
  plt.subplot(311) ; plt.title('betas')
  for i, beta in enumerate(betas): plt.plot(beta, label=i)
  plt.legend()
  plt.subplot(312) ; plt.title('gammas')
  for i, gamma in enumerate(gammas): plt.plot(gamma, label=i)
  plt.legend()
  plt.subplot(313) ; plt.title('loss')
  plt.plot(losses)
  plt.show()
