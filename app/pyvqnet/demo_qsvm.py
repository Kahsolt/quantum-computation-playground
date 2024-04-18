#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/23 

#from sklearn.svm import SVC
from pyqpanda import *
from pyqpanda.Visualization.circuit_draw import *
from pyvqnet.qnn.svm import vqc_qsvm, QuantumKernel_VQNet, gen_vqc_qsvm_data
import matplotlib ; matplotlib.use('TkAgg')

# ref: https://vqnet20-tutorial.readthedocs.io/en/main/rst/qnn.html#svm

assert gen_vqc_qsvm_data
assert QuantumKernel_VQNet
assert vqc_qsvm

train_features, test_features, train_labels, test_labels, samples = gen_vqc_qsvm_data(training_size=100, test_size=10, gap=0.1)
print('train_features.shape:', train_features.shape)
print('test_features.shape:', test_features.shape)
print('train_labels.shape:', train_labels.shape)
print('test_labels.shape:', test_labels.shape)
print('samples.shape:', samples.shape)

''' functional API '''
#quantum_kernel = QuantumKernel_VQNet(n_qbits=2)
#qsvm = SVC(kernel=quantum_kernel.evaluate)
#qsvm.fit(train_features, train_labels)
#score = qsvm.score(test_features, test_labels)
#print(f"quantum kernel classification test score: {score}")

''' objective API '''
qsvm = vqc_qsvm(minibatch_size=40, maxiter=40, rep=3)
qsvm.plot(train_features, test_features, train_labels, test_labels, samples)
pred = qsvm.train(train_features, train_labels)
pred, acc = qsvm.predict(test_features, test_labels)
qsvm.save_thetas(save_dir='qsvm')
