#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/12/29 

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from pyvqnet.tensor import tensor, QTensor
from pyvqnet.nn import Conv2D, ReLu, Dropout, Linear
from pyvqnet.nn import Module
from pyvqnet.optim import Adam
from pyvqnet.nn import CrossEntropyLoss
from pyvqnet.utils.storage import save_parameters, load_parameters

from data import get_mnist

class LeNet(Module):

  def __init__(self):
    super().__init__()

    self.conv1 = Conv2D(1,  8, (5, 5), (2, 2), 'same')
    self.conv2 = Conv2D(8, 16, (5, 5), (2, 2), 'same')
    self.act   = ReLu()
    self.drop  = Dropout(0.5)
    self.fc    = Linear(16 * 7 * 7, 10)
  
  def forward(self, x):
    x = x               # [B, 1, 28, 28]
    x = self.conv1(x)   # [B, 8, 14, 14]
    x = self.act(x)
    x = self.conv2(x)   # [B, 16, 7, 7]
    x = self.act(x)
    x = self.drop(x)
    x = tensor.flatten(x, 1)
    x = self.fc(x)
    return x


def get_correct_count(label:QTensor, pred:QTensor):
  pred  = pred.to_numpy().argmax(axis=-1).astype(np.int32)
  truth = label.to_numpy().astype(np.int32)
  return np.sum(pred == truth)


model = LeNet()
optimizer = Adam(model.parameters(), lr=0.05)
creterion = CrossEntropyLoss()
batch_size = 128
epoch = 5
model_fp = os.path.join('log', 'vq_cnet_minst.model')
stat_fp  = os.path.join('img', 'vq_cnet_minst.png')


def train():
  ''' Load '''
  if os.path.exists(model_fp):
    print(f'>> [Load] from {model_fp}')
    state_dict = load_parameters(model_fp)
    model.load_state_dict(state_dict)
    return

  print('>> [Train]')

  ''' Data '''
  dataloader = get_mnist(is_train=True, batch_size=batch_size, shuffle=True)
  print(f'n_batches: {len(dataloader)}, n_samples: {len(dataloader.dataset)}')
  
  ''' Train '''
  step = 0
  loss_list, acc_list = [], []
  for e in range(epoch):
    total, ok = 0, 0
    model.train()
    for X, Y in dataloader:
      X, Y = QTensor(X.numpy()), QTensor(Y.numpy())
      output = model(X)
      loss = creterion(Y, output)
      optimizer.zero_grad()
      loss.backward()
      optimizer._step()

      step  += 1
      total += batch_size
      ok    += get_correct_count(Y, output)

      if step % 100 == 0:
        loss_list.append(loss.item())
        acc_list.append(ok / total)
        print(f"[Epoch {e+1} / Step {step}], loss: {loss.item():.7f}, accuracy: {ok / total:.3%}")

  ''' Save '''
  os.makedirs(os.path.dirname(model_fp), exist_ok=True)
  save_parameters(model.state_dict(), model_fp)

  os.makedirs(os.path.dirname(stat_fp), exist_ok=True)
  fig, ax1 = plt.subplots()
  ax1.plot(acc_list, c='r', alpha=0.95, label='accuracy')
  ax2 = ax1.twinx()
  ax2.plot(loss_list, c='b', alpha=0.75, label='loss')
  fig.legend()
  fig.tight_layout()
  fig.suptitle('vq_cnet_minst')
  fig.savefig(stat_fp, dpi=400)


def test():
  print('>> [Test]')

  ''' Data '''
  dataloader = get_mnist(is_train=False, batch_size=batch_size, shuffle=False)
  print(f'n_batches: {len(dataloader)}, n_samples: {len(dataloader.dataset)}')

  total, ok = 0, 0
  model.eval()
  for X, Y in dataloader:
    X, Y = QTensor(X.numpy()), QTensor(Y.numpy())
    output = model(X)
    
    total += batch_size
    ok    += get_correct_count(Y, output)

  print(f"test accuracy: {ok / total:.3%}")


def attack(steps=40, eps=0.03, alpha=0.001):
  print('>> [Attack]')

  ''' Data '''
  dataloader = get_mnist(is_train=False, batch_size=batch_size, shuffle=False)
  print(f'n_batches: {len(dataloader)}, n_samples: {len(dataloader.dataset)}')

  total, ok = 0, 0
  model.eval()
  for X, Y in tqdm(dataloader):
    X_orig = X.numpy()
    X_adv = X_orig + np.random.uniform(-eps, eps, size=X.shape)
    Y = QTensor(Y.numpy())

    for _ in range(steps):
      X = QTensor(X_adv)
      X.requires_grad = True
      
      output = model(X)
      loss = creterion(Y, output)
      loss.backward()
      grad = X.grad.to_numpy()

      X_adv = X_adv + np.sign(grad) * alpha
      delta = np.clip(X_adv - X_orig, -eps, eps)
      X_adv = X_orig + delta
    
    total += batch_size
    ok    += get_correct_count(Y, output)

  print(f"remnent accuracy: {ok / total:.3%}")


if __name__ == '__main__':
  train()
  test()
  attack()
