#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/18 

# make qaunt-data though fixed kernel convolution (sum), looks just like a brightess filter...
# NOTE: no trainable parameters!!

from pathlib import Path
from typing import List
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import pyqpanda as pq

IMG_FP = Path(__name__).absolute().parent.parent.parent / 'img' / '147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg'
assert IMG_FP.is_file()

def stats(x:np.ndarray, title='im'):
  print(f'[{title}]')
  print(f'  shape:', x.shape)
  print(f'  min:', x.min())
  print(f'  max:', x.max())
  print(f'  avg:', x.mean())
  print(f'  std:', x.std())

class QCNN_2x2:

  def encode_cir(self, qv, pixels) -> pq.QCircuit:
    cq = pq.QCircuit()
    for i, pix in enumerate(pixels):
      cq << pq.RY(qv[i], np.arctan(pix))
      cq << pq.RZ(qv[i], np.arctan(pix**2))
    return cq

  def entangle_cir(self, qv) -> pq.QCircuit:
    k_size = len(qv)
    cq = pq.QCircuit()
    for i in range(k_size):
      ctr, ctred = i, (i + 1) % k_size
      cq << pq.CNOT(qv[ctr], qv[ctred])
    return cq

  def conv_apply(self, pixels:List[float]) -> float:
    ''' Process a squared 2x2 region of the image with a quantum circuit '''

    k_size = len(pixels)
    qvm = pq.MPSQVM()
    qvm.init_qvm()
    qv = qvm.qAlloc_many(k_size)
    cq = pq.QProg()
    cq << self.encode_cir(qv, np.array(pixels) * np.pi / 2)    # [-pi/2, pi/2] ??
    cq << self.entangle_cir(qv)
    # conv kernel: sum, accumulate all prob on |1>
    result = sum([qvm.prob_run_list(cq, [qv[i]], -1)[-1] for i in range(k_size)])

    qvm.finalize()
    return result

  def forward(self, x:np.ndarray) -> np.ndarray:
    """Convolves the input image with many applications of the same quantum circuit."""

    H, W, C = x.shape
    z = np.zeros([H//2, W//2, C])
    for i in range(0, H, 2):
      for j in range(0, W, 2):
        for k in range(0, C):
          z[i // 2, j // 2, k] = self.conv_apply([
            x[i,     j,     k],
            x[i,     j + 1, k],
            x[i + 1, j,     k],
            x[i + 1, j + 1, k],
          ])
    return z

if __name__ == '__main__':
  img = Image.open(IMG_FP)
  #img = img.convert('L')
  img = img.resize((128, 128), resample=Image.ANTIALIAS)

  im = np.asarray(img, dtype=np.float32) / 255.0
  #im = im * 2 - 1
  if len(im.shape) == 2: im = np.expand_dims(im, -1)

  stats(im, 'im')
  qim = QCNN_2x2().forward(im)
  stats(qim, 'qim')
  qim = qim / qim.max()

  plt.subplot(131) ; plt.title('Resize') ; plt.axis('off') ; plt.imshow(np.asarray(img.resize((64, 64), resample=Image.ANTIALIAS), dtype=np.float32) / 255.0)
  plt.subplot(132) ; plt.title('Raw')    ; plt.axis('off') ; plt.imshow(im)
  plt.subplot(133) ; plt.title('QConv')  ; plt.axis('off') ; plt.imshow(qim)
  plt.show()
