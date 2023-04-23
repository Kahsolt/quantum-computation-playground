#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/14 

# QAE official demo
# ref: https://vqnet20-tutorial.readthedocs.io/en/main/rst/qml_demo.html#id7

import os
import gzip
import urllib.request
import numpy as np
import matplotlib ; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pyvqnet.data import data_generator
from pyvqnet.nn.loss import fidelityLoss
from pyvqnet.qnn.qae import QAElayer
from pyvqnet.optim import Adam

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
  'train_img':'train-images-idx3-ubyte.gz',
  'train_label':'train-labels-idx1-ubyte.gz',
  'test_img':'t10k-images-idx3-ubyte.gz',
  'test_label':'t10k-labels-idx1-ubyte.gz'
}

def _download(dataset_dir,file_name):
  file_path = dataset_dir + '/' + file_name

  if os.path.exists(file_path):
    with gzip.GzipFile(file_path) as f:
      file_path_ungz = file_path[:-3]
      if not os.path.exists(file_path_ungz):
        open(file_path_ungz,'wb').write(f.read())
    return

  print('Downloading ' + file_name + ' ... ')
  urllib.request.urlretrieve(url_base + file_name, file_path)
  if os.path.exists(file_path):
      with gzip.GzipFile(file_path) as f:
        file_path_ungz = file_path[:-3]
        file_path_ungz = file_path_ungz.replace('-idx', '.idx')
        if not os.path.exists(file_path_ungz):
          open(file_path_ungz,'wb').write(f.read())
  print('Done')

def load_mnist(dataset='training_data', digits=np.arange(2), path='data'):
  import os, struct
  from array import array as pyarray

  for v in key_file.values():
    _download(path,v)

  if dataset == 'training_data':
    fname_image = os.path.join(path, 'train-images.idx3-ubyte')
    fname_label = os.path.join(path, 'train-labels.idx1-ubyte')
  elif dataset == 'testing_data':
    fname_image = os.path.join(path, 't10k-images.idx3-ubyte')
    fname_label = os.path.join(path, 't10k-labels.idx1-ubyte')
  else:
    raise ValueError('dataset must be training_data or testing_data')

  flbl = open(fname_label, 'rb')
  magic_nr, size = struct.unpack('>II', flbl.read(8))

  lbl = pyarray('b', flbl.read())
  flbl.close()

  fimg = open(fname_image, 'rb')
  magic_nr, size, rows, cols = struct.unpack('>IIII', fimg.read(16))
  img = pyarray('B', fimg.read())
  fimg.close()

  ind = [k for k in range(size) if lbl[k] in digits]
  N = len(ind)
  images = np.zeros((N, rows, cols))
  labels = np.zeros((N, 1), dtype=int)
  for i in range(len(ind)):
    images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
    labels[i] = lbl[ind[i]]
  return images, labels


def go():
  HW = 2

  x_train, y_train = load_mnist('training_data')
  x_train = x_train / 255
  x_train = x_train.reshape([-1, 1, 28, 28])
  x_train = x_train[:100, :, :, :]
  x_train = np.resize(x_train, [x_train.shape[0], 1, HW, HW])

  x_test, y_test = load_mnist('testing_data')
  x_test = x_test / 255
  x_test = x_test.reshape([-1, 1, 28, 28])
  x_test = x_test[:10, :, :, :]
  x_test = np.resize(x_test, [x_test.shape[0], 1, HW, HW])

  encode_qubits = HW**2
  latent_qubits = 2
  trash_qubits = encode_qubits - latent_qubits
  total_qubits = 1 + trash_qubits + encode_qubits
  print('trash_qubits:', trash_qubits)
  print('total_qubits:', total_qubits)
  model = QAElayer(trash_qubits, total_qubits)
  breakpoint()
  print('param_cnt:', sum([p.size for p in model.parameters()]))

  optimizer = Adam(model.parameters(), lr=0.005)
  creterion = fidelityLoss()

  loss_list = []
  loss_list_test = []
  fidelity_train = []
  fidelity_val = []
  for epoch in range(1, 100):
    # LR sched
    if epoch % 5 == 1: optimizer.lr *= 0.5

    # Train
    running_fidelity_train = 0
    total = 0
    correct = 0
    full_loss = 0
    batch_size = 1

    # x_train: ndarray, [100, 1, 2, 2]
    # y_train: ndarray, [12665, 1]
    model.train()
    for x, y in data_generator(x_train, y_train, batch_size=batch_size, shuffle=True):
      # x: [1, 1, 2, 2]
      # y: [1, 1]

      # [1, 4]
      x = x.reshape((-1, encode_qubits))
      # [1, 7]
      x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]), x), 1)

      optimizer.zero_grad()
      output = model(x)
      loss = creterion(output)
      loss_data = np.array(loss.data)
      loss.backward()
      optimizer._step()

      np_out = np.array(output.data)
      np_output = np.array(output.data, copy=False)
      full_loss += loss_data[0]
      running_fidelity_train += np_out[0]

      correct += sum(np_output.argmax(1) == y.argmax(1))
      total   += batch_size

    loss_output = full_loss / total
    print(f'[train] Epoch: {epoch}, loss: {loss_output.item()}, acc: {correct/total}')
    loss_list.append(loss_output)

    # Test
    running_fidelity_val = 0
    total = 0
    correct = 0
    full_loss = 0
    batch_size = 1

    model.eval()
    for x, y in data_generator(x_test, y_test, batch_size=batch_size, shuffle=True):
      x = x.reshape((-1, encode_qubits))
      x = np.concatenate((np.zeros([batch_size, 1 + trash_qubits]), x), 1)

      output = model(x)
      loss = creterion(output)
      loss_data = np.array(loss.data)
      full_loss += loss_data[0]
      running_fidelity_val += np.array(output.data)[0]

      correct += sum(loss_data.argmax(1) == y.argmax(1))
      total   += batch_size

    loss_output = full_loss / total
    print(f'[valid] epoch: {epoch}, loss: {loss_output.item()}, acc: {correct/total}')
    loss_list_test.append(loss_output)

    fidelity_train.append(running_fidelity_train / 64)
    fidelity_val  .append(running_fidelity_val   / 64)

  plt.plot(loss_list,      color='blue', label='train')
  plt.plot(loss_list_test, color='red',  label='valid')
  plt.title('QAE')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()


if __name__ == '__main__':
  go()
