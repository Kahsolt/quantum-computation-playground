#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/23 

import math
import numpy as np
from pyvqnet.tensor import QTensor, zeros
import pyvqnet.tensor as tensor
import pyqpanda as pq
from sklearn.datasets import make_blobs
import matplotlib ; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ref: https://vqnet20-tutorial.readthedocs.io/en/main/rst/qml_demo.html#quantum-kmeans
# idea: calc the euclid dist between |x> and |y> => do CSWAP test and measure on ancilla bit, more amplitude on |1> means more distance

def get_data(n, k, std):
  return make_blobs(n_samples=n, n_features=2, centers=k, cluster_std=std, random_state=100)

def get_theta(d):
  x, y = d[0], d[1]
  return 2 * math.acos((x.item() + y.item()) / 2.0)

def qkmeans_circuits(x, y):
  ''' pair-wisely measure distance between |x> and |y> '''

  num_qubits = 3
  machine = pq.CPUQVM()
  machine.init_qvm()
  qubits = machine.qAlloc_many(num_qubits)
  cbit   = machine.cAlloc()

  qc = pq.QCircuit() \
    << pq.H(qubits) \
    << pq.U3(qubits[1], get_theta(x), np.pi, np.pi) \
    << pq.U3(qubits[2], get_theta(y), np.pi, np.pi) \
    << pq.SWAP(qubits[1], qubits[2]).control([qubits[0]]) \
    << pq.H(qubits[0])

  prog = pq.QProg() \
    << qc \
    << pq.Measure(qubits[0], cbit)

  n_shot = 1024
  result = machine.run_with_configuration(prog, n_shot)
  return 0.0 if len(result) == 1 else result['1'] / n_shot

def draw_plot(points, centers, label=True):
  points  = np.array(points)
  centers = np.array(centers)
  if not label:
    plt.scatter(points[:,0], points[:,1])
  else:
    plt.scatter(points[:,0], points[:,1], c=centers, cmap='viridis')
  plt.xlim(0, 1)
  plt.ylim(0, 1)

def initialize_centers(points,k):
  return points[np.random.randint(points.shape[0],size=k),:]

def find_nearest_neighbour(points, centroids):
  n = points.shape[0]
  k = centroids.shape[0]

  centers = zeros([n])
  for i in range(n):
    min_dis = 10000
    ind = 0
    for j in range(k):
      temp_dis = qkmeans_circuits(points[i, :], centroids[j, :])
      if temp_dis < min_dis:
        min_dis = temp_dis
        ind = j
    centers[i] = ind

  return centers

def find_centroids(points, centers):
  k = int(tensor.max(centers).item()) + 1
  centroids = tensor.zeros([k, 2])
  for i in range(k):
    cur_i = centers == i

    x = points[:,0][cur_i]
    y = points[:,1][cur_i]
    centroids[i, 0] = tensor.mean(x)
    centroids[i, 1] = tensor.mean(y)

  return centroids

def preprocess(points):
  n = len(points)
  x = 30.0 * np.sqrt(2)
  for i in range(n):
    points[i, :] += 15
    points[i, :] /= x
  return points

def qkmean_run():
  n = 100  # number of data points
  k = 3    # Number of centers
  std = 2  # std of datapoints

  points, o_centers = get_data(n, k, std)    # dataset
  points = preprocess(points)                # Normalize dataset
  centroids = initialize_centers(points, k)  # Intialize centroids

  epoch = 10
  points = QTensor(points)
  centroids = QTensor(centroids)
  plt.subplot(211)
  draw_plot(points.data, o_centers,label=False)

  for _ in range(epoch):
    centers = find_nearest_neighbour(points, centroids)  # find nearest centers
    centroids = find_centroids(points, centers)          # find centroids

  plt.subplot(212)
  draw_plot(points.data, centers.data)
  plt.show()

if __name__ == "__main__":
  qkmean_run()
