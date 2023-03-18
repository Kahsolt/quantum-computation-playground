#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T
from torchvision.utils import make_grid
from sklearnex import patch_sklearn ; patch_sklearn()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

DATA_PATH = 'data'
HW = 28


def get_mnist(is_train=True, batch_size=32, shuffle=False):
  transform = T.Compose([
    #T.Resize((HW, HW), interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
  ])
  dataset    = MNIST(root=DATA_PATH, train=is_train, transform=transform, download=True)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=True, num_workers=0)
  return dataloader


def pca(X: Tensor, n_dim=30) -> Tensor:
  X = X.flatten(start_dim=1)
  X_np = X.cpu().numpy()

  pca = PCA(n_dim, svd_solver='randomized')
  pca.fit(X_np)
  X_pca = pca.transform(X_np)
  print('X_pca.shape:', X_pca.shape)

  plt.clf()
  plt.plot(pca.explained_variance_ratio_)
  plt.show()

  return torch.from_numpy(X_pca)


def binarize(X: Tensor) -> Tensor:
  avg = X.mean(dim=[1, 2, 3], keepdim=True)
  return (X > avg).int()


def imshow(X: Tensor, title='', fp=None):
  grid_X = make_grid(X).permute([1, 2, 0]).detach().cpu().numpy()
  plt.clf()
  plt.axis('off')
  plt.imshow(grid_X)
  plt.suptitle(title)
  plt.tight_layout()
  if fp: plt.savefig(fp, dpi=400)
  else:  plt.show()


if __name__ == '__main__':
  dataloader = get_mnist()
  g = iter(dataloader)

  X, Y = g.next()
  print(X.shape)    # [B=32, C=1, H=28, W=28]
  print(Y.shape)    # [B=32]
  print(X[0])
  print(Y)

  imshow(X)

  X_bin = binarize(X) * 255
  imshow(X_bin)

  X_pca = pca(X)
  X_pca = pca(X_bin)
