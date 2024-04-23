#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/22 

# Create arbitary hand-crafted QNN architectures using VQNet
#   - ref: https://vqnet20-tutorial.readthedocs.io/en/latest/rst/qml_demo.html#id54
# Here'are possible implementation:
#   - VQCLayer (faster speed!): use `VariationalQuantumGate`; define a VQC_wrapper class, override `build_common_circuits()`, `build_vqc_circuits()` and `run()`
#   - QuantumLayer / NoiseQuantumLayer: use common gates; define a process wrapper function, implement both circuit building and results measuring in one run
#   - QuantumLayerV2
# [Timer] 
#   VQC_wrapper: 125.11470079421997
#   QuantumLayer: 171.21107530593872
#   QuantumLayer + ProbsMeasure: 172.05198645591736
#   QuantumLayerV2 (global qvm): 167.12340354919434
#   QuantumLayerV2 (local qvm): 257.7155487537384


import numpy as np
import random ; random.seed(1234)
from time import time

from pyqpanda import *
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.qnn.quantumlayer import VQCLayer, VQC_wrapper, QuantumLayer, QuantumLayerV2
from pyvqnet.qnn.measure import ProbsMeasure
from pyvqnet.nn.loss import CategoricalCrossEntropy
from pyvqnet.optim import sgd

# parity func: f(a, b, c, d) = (a+b+c+d)%2
qvc_train_data = [
  0, 1, 0, 0, 1, 
  0, 1, 0, 1, 0, 
  0, 1, 1, 0, 0, 
  0, 1, 1, 1, 1, 
  1, 0, 0, 0, 1,
  1, 0, 0, 1, 0, 
  1, 0, 1, 0, 0, 
  1, 0, 1, 1, 1, 
  1, 1, 0, 0, 0, 
  1, 1, 0, 1, 1,
  1, 1, 1, 0, 1, 
  1, 1, 1, 1, 0,
]
qvc_test_data = [
  0, 0, 0, 0, 0, 
  0, 0, 0, 1, 1, 
  0, 0, 1, 0, 1, 
  0, 0, 1, 1, 0,
]

def get_data(split):
  datasets = np.array(qvc_train_data if split == "train" else qvc_test_data)
  datasets = datasets.reshape([-1, 5])
  data  = datasets[:, :-1].astype(float)
  label = datasets[:, -1] .astype(int)
  label = np.eye(2)[label].reshape(-1, 2)   # one-hot
  #print('X.shape', data. shape)
  #print('Y.shape', label.shape)
  return data, label

def dataloader(data, label, batch_size, shuffle=True):
  if shuffle:
    for _ in range(len(data)//batch_size):
      random_index = np.random.randint(0, len(data), (batch_size, 1))
      yield data[random_index].reshape(batch_size,-1),label[random_index].reshape(batch_size,-1)
  else:
    for i in range(0,len(data)-batch_size+1,batch_size):
      yield data[i:i+batch_size], label[i:i+batch_size]

def get_accuary(result,label):
  result,label = np.array(result.data), np.array(label.data)
  return np.sum(np.argmax(result,axis=1)==np.argmax(label,1))


use_qmeasure = False    # set False is a bit faster

''' new API: QuantumLayer '''
def qvc_circuits(inputs, weights, qlist, clist, machine):
  def enc_layer(inputs, qubits):
    qc = QCircuit()
    for i in range(len(qubits)):
      if inputs[i] == 1:
        qc << X(qubits[i])
    return qc

  def rot_layer(weights, qubits):
    qc = QCircuit()
    qc << RZ(qubits, weights[0])
    qc << RY(qubits, weights[1])
    qc << RZ(qubits, weights[2])
    return qc

  def ent_layer(qubits):
    qc = QCircuit()
    for i in range(len(qubits)-1):
      qc << CNOT(qubits[i], qubits[i+1])
    qc << CNOT(qubits[len(qubits)-1], qubits[0])
    return qc

  def build_circult(weights, inputs, qubits):
    qc = QCircuit() << enc_layer(inputs, qubits)
    for i in range(weights.shape[0]):
      weights_i = weights[i, :, :]
      for j in range(len(qubits)):
        weights_j = weights_i[j]
        qc << rot_layer(weights_j, qubits[j])
      qc << ent_layer(qubits)
    qc << Z(qubits[0])
    return qc

  weights = weights.reshape([2, 4, 3])
  prog = QProg() << build_circult(weights, inputs, qlist)
  if use_qmeasure:
    prob = ProbsMeasure([0], prog, machine, qlist)
  else:
    prob = machine.prob_run_list(prog, qlist[0])
  return prob

''' new API: QuantumLayerV2 '''
machine = CPUQVM()
machine.init_qvm()
qlist = machine.qAlloc_many(4)

def qvc_circuits_v2(inputs, weights):
  def enc_layer(inputs, qubits):
    qc = QCircuit()
    for i in range(len(qubits)):
      if inputs[i] == 1:
        qc << X(qubits[i])
    return qc

  def rot_layer(weights, qubits):
    qc = QCircuit()
    qc << RZ(qubits, weights[0])
    qc << RY(qubits, weights[1])
    qc << RZ(qubits, weights[2])
    return qc

  def ent_layer(qubits):
    qc = QCircuit()
    for i in range(len(qubits)-1):
      qc << CNOT(qubits[i], qubits[i+1])
    qc << CNOT(qubits[len(qubits)-1], qubits[0])
    return qc

  def build_circult(weights, inputs, qubits):
    qc = QCircuit() << enc_layer(inputs, qubits)
    for i in range(weights.shape[0]):
      weights_i = weights[i, :, :]
      for j in range(len(qubits)):
        weights_j = weights_i[j]
        qc << rot_layer(weights_j, qubits[j])
      qc << ent_layer(qubits)
    qc << Z(qubits[0])
    return qc

  global machine, qlist   # use global avoiding extra cost

  weights = weights.reshape([2, 4, 3])
  prog = QProg() << build_circult(weights, inputs, qlist)
  if use_qmeasure:
    prob = ProbsMeasure([0], prog, machine, qlist)
  else:
    prob = machine.prob_run_list(prog, qlist[0])
  return prob

''' old API: VQCLayer + VQC_wrapper '''
class QVC_runner(VQC_wrapper):

  def build_common_circuits(self, input, qubits):
    qc = QCircuit()
    for i in range(len(qubits)):
      if input[i] == 1:
        qc << X(qubits[i])
    return qc

  def build_vqc_circuits(self, input, weights, machine, qlists, clists):
    def rot_layer(weights, qubits):
      vqc = VariationalQuantumCircuit()
      vqc << VariationalQuantumGate_RZ(qubits, weights[0])
      vqc << VariationalQuantumGate_RY(qubits, weights[1])
      vqc << VariationalQuantumGate_RZ(qubits, weights[2])
      return vqc

    def ent_layer(qubits):
      vqc = VariationalQuantumCircuit()
      for i in range(len(qubits)-1):
        vqc << VariationalQuantumGate_CNOT(qubits[i],qubits[i+1])
      vqc << VariationalQuantumGate_CNOT(qubits[len(qubits)-1], qubits[0])
      return vqc
    
    def build_circult(weights, inputs, qubits):
      vqc = VariationalQuantumCircuit()
      for i in range(2):
        weights_i = weights[i,:,:]
        for j in range(len(qubits)):
          weights_j = weights_i[j]
          vqc << rot_layer(weights_j, qubits[j])
        vqc << ent_layer(qubits)
      vqc << VariationalQuantumGate_Z(qubits[0])  # pauli z(0)
      return vqc

    weights = weights.reshape([2, 4, 3])
    return build_circult(weights, input, qlists)

  def run(self, vqc, inputs, machine, qlists, clists):
    vqc_all = VariationalQuantumCircuit()
    # basic encode, why the fuck this `build_common_circuits()` is not auto called??
    vqc_all << self.build_common_circuits(inputs, qlists)
    # vqc transfrom, vqc is auto obtained from `build_vqc_circuits()`
    vqc_all << vqc
    prog = QProg() << vqc_all.feed()
    prob = machine.prob_run_list(prog, qlists[0])
    return prob


def go(method, perf_cnt=True):
  if method is VQC_wrapper:
    model = VQCLayer(QVC_runner(), 24, "cpu", 4)
  elif method is QuantumLayer:
    model = QuantumLayer(qvc_circuits, 24, "cpu", 4)
  elif method is QuantumLayerV2:
    model = QuantumLayerV2(qvc_circuits_v2, 24)

  batch_size = 4
  epoch = 1000 if perf_cnt else 20
  optimizer = sgd.SGD(model.parameters(), lr=0.5)
  creterion = CategoricalCrossEntropy()

  if not perf_cnt: print("start training..............")
  datas, labels = get_data('train')
  model.train()
  for i in range(epoch):
    count = 0
    accuary = 0
    sum_loss = 0

    for data, label in dataloader(datas, labels, batch_size):
      data, label = QTensor(data), QTensor(label)
      
      optimizer.zero_grad()
      result = model(data)
      loss = creterion(label, result)
      loss.backward()
      optimizer._step()
      
      sum_loss += loss.item()
      count += batch_size
      accuary += get_accuary(result,label)

    if not perf_cnt: print(f"[Epoch-{i}] loss:{sum_loss/count}, accuray:{accuary/count}")

  if not perf_cnt: print("start testing..............")
  batch_size = 1
  count = 0
  accuary = 0
  sum_loss = 0

  datas, labels = get_data("test")
  model.eval()
  for data, label in dataloader(datas, labels, batch_size, False):
    data = QTensor(data)
    result = model(data)
    sum_loss += creterion(label, result)
    count += batch_size
    accuary += get_accuary(result, label)

  if not perf_cnt: print(f"[Test] loss:{sum_loss/count}, accuray:{accuary/count}")


if __name__=="__main__":
  t = time()
  go(VQC_wrapper)
  print(f'[Timer] VQC_wrapper: {time() - t}')

  t = time()
  go(QuantumLayer)
  print(f'[Timer] QuantumLayer: {time() - t}')

  t = time()
  go(QuantumLayerV2)
  print(f'[Timer] QuantumLayerV2: {time() - t}')
