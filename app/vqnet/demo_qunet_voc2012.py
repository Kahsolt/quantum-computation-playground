#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/17 '

# ref: https://vqnet20-tutorial.readthedocs.io/en/main/rst/qml_demo.html#id19
# 仅量子数据：像素值QConv[k=2]转概率幅

import os
import numpy as np

from pyqpanda import *
from pyvqnet.tensor import tensor
from pyvqnet.tensor.tensor import QTensor
from pyvqnet.nn.module import Module
from pyvqnet.nn.conv import Conv2D, ConvT2D
from pyvqnet.nn import activation as F
from pyvqnet.nn.batch_norm import BatchNorm2d
from pyvqnet.nn.loss import BinaryCrossEntropy
from pyvqnet.optim.adam import Adam
from pyvqnet.utils.storage import save_parameters

import cv2
import matplotlib ; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from qcnn_2x2 import QCNN_2x2


class PreprocessingData:

  def __init__(self, path):
    self.path = path
    self.x_data = []
    self.y_label = []

  def processing(self):
    list_path = os.listdir((self.path + "/images"))
    for i in range(len(list_path)):
      temp_data = cv2.imread(self.path + "/images" + '/' + list_path[i], cv2.IMREAD_COLOR)
      temp_data = cv2.resize(temp_data, (128, 128))
      grayimg = cv2.cvtColor(temp_data, cv2.COLOR_BGR2GRAY)
      temp_data = grayimg.reshape(temp_data.shape[0], temp_data.shape[0], 1)
      self.x_data.append(temp_data)

      label_data = cv2.imread(self.path+"/labels" + '/' +list_path[i].split(".")[0] + ".png", cv2.IMREAD_COLOR)
      label_data = cv2.resize(label_data, (128, 128))
      label_data = cv2.cvtColor(label_data, cv2.COLOR_BGR2GRAY)
      label_data = label_data.reshape(label_data.shape[0], label_data.shape[0], 1)
      self.y_label.append(label_data)
    return self.x_data, self.y_label

  def read(self):
    self.x_data, self.y_label = self.processing()
    x_data  = np.array(self.x_data)
    y_label = np.array(self.y_label)
    return x_data, y_label

class MyDataset():

  def __init__(self, x_data, x_label):
    self.data  = x_data
    self.label = x_label

  def __getitem__(self, item):
    img, target = self.data[item], self.label[item]
    img    = np.uint8(img)   .transpose(2, 0, 1)
    target = np.uint8(target).transpose(2, 0, 1)
    return img, target

  def __len__(self):
    return len(self.data)

def quantum_data_preprocessing(images):
  return np.asarray([QCNN_2x2().forward(img) for img in images])


class DownsampleLayer(Module):

  def __init__(self, in_ch, out_ch):
    super().__init__()

    self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding="same")
    self.bn1 = BatchNorm2d(out_ch)
    self.conv2 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(1, 1), padding="same")
    self.bn2 = BatchNorm2d(out_ch)
    self.conv3 = Conv2D(input_channels=out_ch, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2), padding="same")
    self.bn3 = BatchNorm2d(out_ch)
    self.relu = F.ReLu()
    
  def forward(self, x):
    """
    :param x:
    :return: out(Output to deep)，out_2(enter to next level)，
    """
    x1 = self.conv1(x)
    x2 = self.bn1(x1)
    x3 = self.relu(x2)
    x4 = self.conv2(x3)
    x5 = self.bn2(x4)
    out = self.relu(x5)
    x6 = self.conv3(out)
    x7 = self.bn3(x6)
    out_2 = self.relu(x7)
    return out, out_2

class UpSampleLayer(Module):
   
  def __init__(self, in_ch, out_ch):
    super(UpSampleLayer, self).__init__()
    
    self.conv1 = Conv2D(input_channels=in_ch, output_channels=out_ch*2, kernel_size=(3, 3), stride=(1, 1), padding="same")
    self.BatchNorm2d1 = BatchNorm2d(out_ch*2)
    self.conv2 = Conv2D(input_channels=out_ch*2, output_channels=out_ch*2, kernel_size=(3, 3), stride=(1, 1), padding="same")
    self.BatchNorm2d2 = BatchNorm2d(out_ch*2)
    self.conv3 = ConvT2D(input_channels=out_ch*2, output_channels=out_ch, kernel_size=(3, 3), stride=(2, 2), padding="same")
    self.BatchNorm2d3 = BatchNorm2d(out_ch)
    self.relu = F.ReLu()

  def forward(self, x):
    '''
    :param x: input conv layer
    :param out: connect with UpsampleLayer
    :return:
    '''
    x = self.conv1(x)
    x = self.BatchNorm2d1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.BatchNorm2d2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.BatchNorm2d3(x)
    x_out = self.relu(x)
    return x_out

class UNet(Module):
  def __init__(self):
    super(UNet, self).__init__()
    
    out_channels = [2 ** (i + 4) for i in range(5)]

    # DownSampleLayer
    self.d1 = DownsampleLayer(1, out_channels[0])  # 3-64
    self.d2 = DownsampleLayer(out_channels[0], out_channels[1])  # 64-128
    self.d3 = DownsampleLayer(out_channels[1], out_channels[2])  # 128-256
    self.d4 = DownsampleLayer(out_channels[2], out_channels[3])  # 256-512
    # UpSampleLayer
    self.u1 = UpSampleLayer(out_channels[3], out_channels[3])  # 512-1024-512
    self.u2 = UpSampleLayer(out_channels[4], out_channels[2])  # 1024-512-256
    self.u3 = UpSampleLayer(out_channels[3], out_channels[1])  # 512-256-128
    self.u4 = UpSampleLayer(out_channels[2], out_channels[0])  # 256-128-64

    # output
    self.conv1 = Conv2D(input_channels=out_channels[1], output_channels=out_channels[0], kernel_size=(3, 3), stride=(1, 1), padding="same")
    self.bn1 = BatchNorm2d(out_channels[0])
    self.conv2 = Conv2D(input_channels=out_channels[0], output_channels=out_channels[0], kernel_size=(3, 3), stride=(1, 1), padding="same")
    self.bn2 = BatchNorm2d(out_channels[0])
    self.conv3 = Conv2D(input_channels=out_channels[0], output_channels=1, kernel_size=(3, 3), stride=(1, 1), padding="same")
    self.relu = F.ReLu()
    self.sigmoid = F.Sigmoid()

  def forward(self, x):
    out_1, out1 = self.d1(x)
    out_2, out2 = self.d2(out1)
    out_3, out3 = self.d3(out2)
    out_4, out4 = self.d4(out3)

    out5 = self.u1(out4)
    cat_out5 = tensor.concatenate([out5, out_4], axis=1)
    out6 = self.u2(cat_out5)
    cat_out6 = tensor.concatenate([out6, out_3], axis=1)
    out7 = self.u3(cat_out6)
    cat_out7 = tensor.concatenate([out7, out_2], axis=1)
    out8 = self.u4(cat_out7)
    cat_out8 = tensor.concatenate([out8, out_1], axis=1)
    
    out = self.conv1(cat_out8)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv3(out)
    out = self.sigmoid(out)
    return out


def train():
  if not os.path.exists("./result"): os.makedirs("./result")
  if not os.path.exists("./Intermediate_results"): os.makedirs("./Intermediate_results")

  # prepare train/test data and label
  train_x, train_y = PreprocessingData('training_data').read()
  test_x,  test_y  = PreprocessingData('testing_data') .read()
  print('train: ', train_x.shape, '\ntest: ', test_x.shape)
  print('train: ', train_y.shape, '\ntest: ', test_y.shape)
  train_x = train_x / 255
  test_x  = test_x  / 255

  # use quantum encoder to preprocess data
  PREPROCESS = True

  if PREPROCESS == True:
    print("Quantum pre-processing of train images:")
    q_train_images = quantum_data_preprocessing(train_x)
    q_test_images  = quantum_data_preprocessing(test_x)
    q_train_label  = quantum_data_preprocessing(train_y)
    q_test_label   = quantum_data_preprocessing(test_y)

    # Save pre-processed images
    print('Quantum Data Saving...')
    np.save("./result/q_train.npy",       q_train_images)
    np.save("./result/q_test.npy",        q_test_images)
    np.save("./result/q_train_label.npy", q_train_label)
    np.save("./result/q_test_label.npy",  q_test_label)
    print('Quantum Data Saving Over!')

  # loading quantum data
  SAVE_PATH = "./result/"
  train_x = np.load(SAVE_PATH + "q_train.npy")      .astype(np.uint8)
  train_y = np.load(SAVE_PATH + "q_train_label.npy").astype(np.uint8)
  test_x  = np.load(SAVE_PATH + "q_test.npy")       .astype(np.uint8)
  test_y  = np.load(SAVE_PATH + "q_test_label.npy") .astype(np.uint8)

  trainset = MyDataset(train_x, train_y)
  testset  = MyDataset(test_x, test_y)
  x_train = []
  y_label = []
  model = UNet()
  optimizer = Adam(model.parameters(), lr=0.01)
  loss_func = BinaryCrossEntropy()
  epochs = 200

  loss_list = []
  SAVE_FLAG = True
  temp_loss = 0
  file = open("./result/result.txt", 'w').close()
  for epoch in range(1, epochs):
    total_loss = []
    model.train()
    for i, (x, y) in enumerate(trainset):
      x_img_Qtensor = tensor.unsqueeze(QTensor(x), 0)
      y_img_Qtensor = tensor.unsqueeze(QTensor(y), 0)

      optimizer.zero_grad()
      img_out = model(x_img_Qtensor)

      print(f"=========={epoch}==================")
      loss = loss_func(y_img_Qtensor, img_out)  # target output
      if i == 1:
          plt.figure()
          plt.subplot(1, 2, 1)
          plt.title("predict")
          img_out_tensor = tensor.squeeze(img_out, 0)

          if matplotlib.__version__ >= '3.4.2':
            plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]))
          else:
            plt.imshow(np.array(img_out_tensor.data).transpose([1, 2, 0]).squeeze(2))
          plt.subplot(1, 2, 2)
          plt.title("label")
          y_img_tensor = tensor.squeeze(y_img_Qtensor, 0)
          if matplotlib.__version__ >= '3.4.2':
            plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]))
          else:
            plt.imshow(np.array(y_img_tensor.data).transpose([1, 2, 0]).squeeze(2))

          plt.savefig("./Intermediate_results/" + str(epoch) + "_" + str(i) + ".jpg")

      loss_data = np.array(loss.data)
      print("{} - {} loss_data: {}".format(epoch, i, loss_data))
      loss.backward()
      optimizer._step()
      total_loss.append(loss_data)

    loss_list.append(np.sum(total_loss) / len(total_loss))
    out_read = open("./result/result.txt", 'a')
    out_read.write(str(loss_list[-1]))
    out_read.write(str("\n"))
    out_read.close()
    print("{:.0f} loss is : {:.10f}".format(epoch, loss_list[-1]))

    if SAVE_FLAG:
      temp_loss = loss_list[-1]
      save_parameters(model.state_dict(), "./result/Q-Unet_End.model")
      SAVE_FLAG = False
    else:
      if temp_loss > loss_list[-1]:
        temp_loss = loss_list[-1]
        save_parameters(model.state_dict(), "./result/Q-Unet_End.model")


if __name__ == '__main__':
  imgs = np.random.normal(size=[4, 128, 128, 1])
  qimgs = np.asarray([QCNN_2x2().forward(img) for img in imgs])
  print(qimgs.shape)
  print(qimgs.dtype)
