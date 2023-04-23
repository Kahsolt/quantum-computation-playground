#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/17 

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pyqpanda as pq
from pyqpanda.Visualization.draw_probability_map import draw_probaility
from pyvqnet.qnn.qgan.qgan_utils import QGANAPI

# ref: https://vqnet20-tutorial.readthedocs.io/en/main/rst/qnn.html#qgan
assert QGANAPI
QGANAPI.train
QGANAPI.eval
QGANAPI.eval_metric
QGANAPI.get_circuits_with_trained_param
QGANAPI.get_trained_quantum_parameters
QGANAPI.load_param_and_eval


num_of_qubits = 3  # paper config
rep = 1

number_of_data = 10000
mu    = 1
sigma = 1
real_data = np.random.lognormal(mean=mu, sigma=sigma, size=number_of_data)
print('real_data.shape:', real_data.shape)    # [N]

# intial
save_dir = None
qgan_model = QGANAPI(
  real_data,
  # numpy generated data distribution, 1 - dim.
  num_of_qubits,
  batch_size=2000,
  num_epochs=1000,
  q_g_cir=None,
  bounds=[0, 2**num_of_qubits-1],
  reps=rep,
  metric="kl",
  tol_rel_ent=0.01,
  if_save_param_dir=save_dir
)

# train
qgan_model.train()  # train qgan
# show probability distribution function of generated distribution and real distribution
qgan_model.eval(real_data)  #draw pdf

# get trained quantum parameters
param = qgan_model.get_trained_quantum_parameters()
print(f"trained param {param}")

#load saved parameters files
if save_dir is not None:
  path = os.path.join(save_dir, "trained_qgan_param.pkl")
  with open(path, "rb") as file:
    t3 = pickle.load(file)
    print(t3.keys())
  param = t3["quantum_parameters"]
  print(f"trained param {param}")

#show probability distribution function of generated distribution and real distribution
qgan_model.load_param_and_eval(param)

#calculate metric
print('kl:', qgan_model.eval_metric(param, "kl"))

#get generator quantum circuit
machine = pq.CPUQVM()
machine.init_qvm()
qubits = machine.qAlloc_many(num_of_qubits)
qpanda_cir = qgan_model.get_circuits_with_trained_param(qubits)
print(qpanda_cir)

prog = pq.QProg() << qpanda_cir
machine.directly_run(prog)

cv = machine.cAlloc_many(num_of_qubits)
prog << pq.measure_all(qubits, cv)
print(machine.get_qstate())
results = machine.run_with_configuration(prog, cv, shot=1000)
#draw_probaility(results)
plt.show()
