#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/06 

from typing import Tuple, Generator

import numpy as np
import matplotlib.pyplot as plt

import torch
import paddle
import paddle_quantum ; paddle_quantum.set_backend('state_vector')
from paddle_quantum.ansatz import Circuit
from paddle_quantum.state import State, to_state, random_state, zero_state
from paddle_quantum.linalg import dagger
import paddle_quantum.gate.functional as F

from data import get_mnist, binarize

N_QUBIT = 10    # Qubit数, 10分类
N_LAYER = 3     # QNN层数
EPOCH   = 5
BS      = 32
LR      = 0.2   # 学习率


''' Model & Optimizer '''
def make_circuit(n_qubit, n_layer) -> Circuit:
  cir = Circuit(n_qubit)
  for _ in range(n_layer):
    for n in range(n_qubit):
      cir.rx(n)
      cir.rz(n)
    for n in range(n_qubit - 1):
      cir.cnot([n, n + 1])
  return cir

cir = make_circuit(N_QUBIT, N_LAYER)
print(cir)
param_cnt = sum([p.numel() for p in cir.parameters()]).item()
print(f'  param_cnt: {param_cnt}')

optimizer = paddle.optimizer.Adam(learning_rate=LR, parameters=cir.parameters())


''' Measure '''
def measure(cir: Circuit, phi: State) -> int:
  final_state = cir(phi).data  # 网络输出
  psi_target = psi.data        # 目标值
  # 投影测量？
  breakpoint()
  inner = paddle.matmul(dagger(final_state), psi_target)
  loss = -paddle.real(paddle.matmul(dagger(inner), inner))
  return loss


''' Loss '''
def get_loss(cir: Circuit, phi: State, psi: State) -> paddle.Tensor:
  final_state = cir(phi).data  # 网络输出
  psi_target = psi.data        # 目标值
  # 投影测量？
  breakpoint()
  inner = paddle.matmul(dagger(final_state), psi_target)
  loss = -paddle.real(paddle.matmul(dagger(inner), inner))
  return loss


''' Data '''
def gen_input_batch(X: torch.Tensor, Y: torch.Tensor) -> Generator[Tuple[State, State], None, None]:
  X = binarize(X).cpu().numpy()
  Y = Y          .cpu().numpy()

  def encode_x(x: np.array):
    v = np.zeros([2 ** N_QUBIT])
    v[:len(x)] = x
    v /= np.linalg.norm(v, axis=-1)   # 态矢应归一化
    return to_state(v.astype(np.complex64), num_qubits=N_QUBIT)

  def encode_y(y: int) -> np.array:
    v = zero_state(num_qubits=N_QUBIT)
    return F.x(v, qubit_idx=y, dtype=np.complex64, backend='state_vector')
  
  for x, y in zip(X, Y):
    breakpoint()
    phi = encode_x(x.flatten())     # binary vector [D=768]
    psi = encode_y(y.item())        # scalar
    yield phi, psi

trainloader = get_mnist(is_train=True, batch_size=BS, shuffle=True)
testloader  = get_mnist(is_train=False, batch_size=1)


''' Train '''
loss_list = []
step = 0
for e in range(EPOCH):
  for X, Y in trainloader:
    losses = []
    for psi, phi in gen_input_batch(X, Y):
      loss = get_loss(cir, psi, phi)
      losses.append(loss)

    loss = sum(losses) / BS
    loss.backward()
    optimizer.minimize(loss)
    optimizer.clear_grad()

    loss_list.append(loss.numpy())

    step += 1
    if step % 10 == 0:
      print(f'step: {step}, loss: {loss.numpy():.4f}')


''' Plot '''
if True:
  plt.clf()
  plt.plot(range(step), loss_list, alpha=0.7, marker='', linestyle='-', color='r')
  plt.xlabel('steps')
  plt.ylabel('loss')
  plt.legend(labels=['training loss'], loc='best')
  plt.show()


''' Model (weights) '''
theta_final = [p.numpy() for p in cir.parameters()]
print('Weights:')
print(theta_final)
print()
print(cir)


''' Test '''
total, correct = 0, 0
for X, Y in testloader:
  for psi, phi in gen_input_batch(X, Y):
    phi_hat = measure(cir, psi)
    
    total   += 1
    correct += int(phi.item() == phi_hat)

print(f'Accuracy: {correct / total:.3%}')
