#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

import pyqpanda as pq
from pyvqnet.tensor import QTensor
from pyvqnet.qnn.qdrl.vqnet_model import qdrl_circuit
from pyvqnet.qnn.quantumlayer import QuantumLayer
from pyvqnet.optim import adam
from pyvqnet.nn.loss import BinaryCrossEntropy, CategoricalCrossEntropy, SoftmaxCrossEntropy, CrossEntropyLoss
from pyvqnet.nn.module import Module


# 待训练参数个数
param_num = 9
# 量子计算模块量子比特数
qbit_num  = 1
class Model(Module):
  def __init__(self):
    super(Model, self).__init__()

    def qdrl_circuit(input, weights, qlist, clist, machine):
      x1     = input  .squeeze()    # [3]
      param1 = weights.squeeze()    # [9]

      circult = pq.QCircuit()
      if 'layer 0':
        circult.insert(pq.RZ(qlist[0], x1[0]))    # RZ编码横坐标x到旋转角度
        circult.insert(pq.RY(qlist[0], x1[1]))    # RY编码纵坐标y到旋转角度
        #circult.insert(pq.RZ(qlist[0], x1[2]))
        circult.insert(pq.RZ(qlist[0], param1[0]))
        circult.insert(pq.RY(qlist[0], param1[1]))
        #circult.insert(pq.RZ(qlist[0], param1[2]))

      if 'layer 1':
        circult.insert(pq.RZ(qlist[0], x1[0]))
        circult.insert(pq.RY(qlist[0], x1[1]))
        #circult.insert(pq.RZ(qlist[0], x1[2]))
        circult.insert(pq.RZ(qlist[0], param1[3]))
        circult.insert(pq.RY(qlist[0], param1[4]))
        #circult.insert(pq.RZ(qlist[0], param1[5]))
        
      if 'layer 2':
        circult.insert(pq.RZ(qlist[0], x1[0]))
        circult.insert(pq.RY(qlist[0], x1[1]))
        #circult.insert(pq.RZ(qlist[0], x1[2]))
        circult.insert(pq.RZ(qlist[0], param1[6]))
        circult.insert(pq.RY(qlist[0], param1[7]))
        #circult.insert(pq.RZ(qlist[0], param1[8]))

      prog = pq.QProg() << circult
      #print(prog)

      prob = machine.prob_run_dict(prog, qlist, -1)
      prob = list(prob.values())    # => 概率分布列
      return prob
      
    self.pqc = QuantumLayer(qdrl_circuit, param_num, 'cpu', qbit_num)

  def forward(self, x):
    return self.pqc(x)

# 随机产生待训练数据的函数
# 单位正方形内投点，单位圆内标记为1，之外标记为0
def circle(samples:int, rads=np.sqrt(2/np.pi)):
  data_x, data_y = [], []
  for i in range(samples):
    x = 2 * np.random.rand(2) - 1
    y = [0, 1]
    if np.linalg.norm(x) < rads:
      y = [1, 0]
    data_x.append(x)
    data_y.append(y)
  return np.array(data_x), np.array(data_y)

def circle_grid(cuts:int, rads=np.sqrt(2/np.pi)):
  data_x, data_y = [], []
  for i in np.linspace(-1, 1, cuts):
    for j in np.linspace(-1, 1, cuts):
      x = [i, j]
      if np.linalg.norm(x) < rads:
        y = [1, 0]
      else:
        y = [0, 1]
      data_x.append(x)
      data_y.append(y)
  return np.array(data_x), np.array(data_y)

def cross(samples:int) :
  data_x, data_y = [], []
  for i in range(samples):
    x = 2 * np.random.rand(2) - 1
    y = [0, 1]
    if abs(x[0]) < abs(x[1]):
      y = [1, 0]
    data_x.append(x)
    data_y.append(y)
  return np.array(data_x), np.array(data_y)

def cross_grid(cuts:int):
  data_x, data_y = [], []
  for i in np.linspace(-1, 1, cuts):
    for j in np.linspace(-1, 1, cuts):
      x = [i, j]
      if abs(i) < abs(j):
        y = [1, 0]
      else:
        y = [0, 1]
      data_x.append(x)
      data_y.append(y)
  return np.array(data_x), np.array(data_y)

def gen_batch(x_data, label, batch_size):
  for i in range(0, x_data.shape[0] - batch_size + 1, batch_size):
    idxs = slice(i, i + batch_size)
    yield x_data[idxs], label[idxs]

def get_correct_count(pred, label):
  pred  = np.argmax(np.array(pred .data), axis=1)
  truth = np.argmax(np.array(label.data), axis=1)
  return np.sum(pred == truth)


model = Model()
optimizer = adam.Adam(model.parameters(), lr=0.6)
creterion = BinaryCrossEntropy()
#creterion = CategoricalCrossEntropy()
#creterion = SoftmaxCrossEntropy()
#creterion = CrossEntropyLoss()


def train(epoch=10, batch_size=32):
  print("start training...........")
  x_train, y_train = dataset(500)
  x_train = np.hstack((x_train, np.zeros((x_train.shape[0], 1))))
  print(x_train.shape)  # [N, D=3], 有一个辅助比特？
  print(x_train.shape)  # [N, NC=2]

  total, ok = 0, 0
  model.train()
  imgs = []
  for i in range(epoch):
    losses = 0
    for Xt, Yt in gen_batch(x_train, y_train, batch_size):
      X, Y = QTensor(Xt), QTensor(Yt)
      output = model(X)

      loss = creterion(Y, output)
      optimizer.zero_grad()
      loss.backward()
      optimizer._step()

      losses += loss.item()
      total  += batch_size
      ok     += get_correct_count(output, Y)
    
    print(f"[Epoch {i}], loss: {losses / total:.7f}, accuracy: {ok / total:.3%}")

    imgs.append(show_grid(batch_size=100, title=f'epoch-{i}, acc={ok / total:.3%}', show=False))

  clip = ImageSequenceClip(imgs, fps=12)
  clip.write_gif(f'img/qnet_clf_{expname}.gif')


def test(batch_size=1):
  print("start eval...................")

  x_test, y_test = dataset(500)
  x_test = np.hstack((x_test, np.zeros((x_test.shape[0], 1))))
  print(x_test.shape)
  print(y_test.shape)

  total, ok = 0, 0
  model.eval()
  for X, Y in gen_batch(x_test, y_test, batch_size):
    X, Y = QTensor(X), QTensor(Y)
    output = model(X)
    
    total += batch_size
    ok    += get_correct_count(output, Y)

  print(f"test accuracy: {ok / total:.3%}")


def show_grid(batch_size=100, title=None, show=True):
  print("start grid...................")

  x_test, y_test = dataset_grid(100)
  x_test = np.hstack((x_test, np.zeros((x_test.shape[0], 1))))
  print(x_test.shape)
  print(y_test.shape)

  model.eval()
  preds = []
  for X, Y in gen_batch(x_test, y_test, batch_size):
    X, Y = QTensor(X), QTensor(Y)
    output = model(X)
    preds.append(output.to_numpy().argmax(-1))
  preds = np.concatenate(preds, axis=0)

  plt.clf()
  plt.subplot(121)
  plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test.argmax(-1), cmap='bwr')
  plt.title('truth')
  plt.subplot(122)
  plt.scatter(x_test[:, 0], x_test[:, 1], c=preds, cmap='bwr')
  plt.title('pred')
  plt.suptitle(title)
  plt.tight_layout()

  if show:
    plt.show()
  else:
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    return np.asarray(Image.open(img_buf))


def pgd_attack(steps=40, alpha=0.01):
  x_test, y_test = dataset(256)
  x_test = np.hstack((x_test, np.zeros((x_test.shape[0], 1))))
  print(x_test.shape)
  print(y_test.shape)

  Y = QTensor(y_test)
  imgs = []
  for i in range(steps):
    X = QTensor(x_test)
    X.requires_grad = True

    output = model(X)
    preds = output.to_numpy().argmax(-1)
    acc = sum(preds == y_test.argmax(-1)) / X.shape[0]
    print(f'step: {i}, acc: {acc:.3%}')

    loss = creterion(Y, output)
    loss.backward()
    grad = X.grad.to_numpy()
    #print('grad:', grad.max(), grad.min())

    # NOTE: 不知为何在这个框架里梯度上升反而要用减号🤔
    x_test = X.to_numpy() - np.sign(grad) * alpha

    plt.clf()
    plt.scatter(x_test[:, 0], x_test[:, 1], c=preds, cmap='bwr')
    plt.title(f'step-{i}, acc={acc:.3%}')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    imgs.append(np.asarray(Image.open(img_buf)))

  clip = ImageSequenceClip(imgs, fps=12)
  clip.write_gif('img/pgd.gif')


if __name__ == '__main__':
  #expname = 'circle'
  expname = 'cross'

  dataset      = globals().get(expname)
  dataset_grid = globals().get(f'{expname}_grid')

  train(epoch=100)
  test()
  #show_grid()
  #pgd_attack()
