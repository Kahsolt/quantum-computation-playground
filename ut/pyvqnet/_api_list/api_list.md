# VQ-Net API list

    PyTorch-like API

----

### Tensor

```python
# misc
pyvqnet.tensor.to_tensor
pyvqnet.tensor.QTensor
pyvqnet.tensor.CoreTensor

# init
pyvqnet.tensor.randn
pyvqnet.tensor.randu
pyvqnet.tensor.zeros
pyvqnet.tensor.zeros_like
pyvqnet.tensor.ones
pyvqnet.tensor.ones_like
pyvqnet.tensor.full
pyvqnet.tensor.full_like
pyvqnet.tensor.empty
pyvqnet.tensor.eye
pyvqnet.tensor.arange
pyvqnet.tensor.linspace
pyvqnet.tensor.logspace

# shape
pyvqnet.tensor.reshape
pyvqnet.tensor.squeeze
pyvqnet.tensor.unsqueeze
pyvqnet.tensor.permute
pyvqnet.tensor.swapaxis
pyvqnet.tensor.flip
pyvqnet.tensor.split
pyvqnet.tensor.chunk
pyvqnet.tensor.concatenate
pyvqnet.tensor.tile

# pair-wise
pyvqnet.tensor.masked_fill
pyvqnet.tensor.minimum
pyvqnet.tensor.maximum
pyvqnet.tensor.broadcast
pyvqnet.tensor.broadcast_to

# index
pyvqnet.tensor.nonzero
pyvqnet.tensor.where
pyvqnet.tensor.select
pyvqnet.tensor.set_select

# arithmetic
pyvqnet.tensor.neg
pyvqnet.tensor.sign
pyvqnet.tensor.abs
pyvqnet.tensor.matmul
pyvqnet.tensor.divide
pyvqnet.tensor.reciprocal
pyvqnet.tensor.power
pyvqnet.tensor.log
pyvqnet.tensor.log_softmax
pyvqnet.tensor.square
pyvqnet.tensor.sqrt
pyvqnet.tensor.round
pyvqnet.tensor.floor
pyvqnet.tensor.ceil
pyvqnet.tensor.clip

# matrix
pyvqnet.tensor.diag
pyvqnet.tensor.trace
pyvqnet.tensor.tril
pyvqnet.tensor.triu
pyvqnet.tensor.frobenius_norm

# triangle
pyvqnet.tensor.sin
pyvqnet.tensor.cos
pyvqnet.tensor.tan
pyvqnet.tensor.sinh
pyvqnet.tensor.cosh
pyvqnet.tensor.tanh
pyvqnet.tensor.asin
pyvqnet.tensor.acos
pyvqnet.tensor.atan

# stats
pyvqnet.tensor.min
pyvqnet.tensor.max
pyvqnet.tensor.std
pyvqnet.tensor.mean
pyvqnet.tensor.median
pyvqnet.tensor.sums
pyvqnet.tensor.cumsum
pyvqnet.tensor.sort
pyvqnet.tensor.argsort
pyvqnet.tensor.topK
pyvqnet.tensor.argtopK

# logical
pyvqnet.tensor.not_equal
pyvqnet.tensor.logical_and
pyvqnet.tensor.logical_or
pyvqnet.tensor.logical_xor
pyvqnet.tensor.logical_not
pyvqnet.tensor.less
pyvqnet.tensor.less_equal
pyvqnet.tensor.greater
pyvqnet.tensor.greater_equal
pyvqnet.tensor.isnan
pyvqnet.tensor.isinf
pyvqnet.tensor.isposinf
pyvqnet.tensor.isneginf
pyvqnet.tensor.isfinite

# unknown (maybe inner API)
pyvqnet.tensor.SplitBackward
pyvqnet.tensor.maybe_wrap_dim
pyvqnet.tensor.var_match_shape
pyvqnet.tensor.get_bdshape_for_slice
pyvqnet.tensor.get_prev
pyvqnet.tensor.backprop
pyvqnet.tensor.toposort
pyvqnet.tensor.helper_function_bmm
pyvqnet.tensor.helper_function_matmul2d
pyvqnet.tensor.helper_function_matmul4d
```

### NN

```python
# module
pyvqnet.nn.Module
pyvqnet.nn.Parameter

# linear
pyvqnet.nn.Linear
pyvqnet.nn.Dropout
pyvqnet.nn.Embedding

# conv
pyvqnet.nn.Conv1D
pyvqnet.nn.Conv2D
pyvqnet.nn.ConvT2D
pyvqnet.nn.Self_Conv_Attention

# pooling
pyvqnet.nn.AvgPool1D
pyvqnet.nn.AvgPool2D
pyvqnet.nn.MaxPool1D
pyvqnet.nn.MaxPool2D

# rnn
pyvqnet.nn.RNN
pyvqnet.nn.LSTM
pyvqnet.nn.GRU
pyvqnet.nn.rnn.reset_layer_params
pyvqnet.nn.rnn.reset_params_names
pyvqnet.nn.rnn.reset_zeros

# activation
pyvqnet.nn.ReLu
pyvqnet.nn.LeakyReLu
pyvqnet.nn.ELU
pyvqnet.nn.Tanh
pyvqnet.nn.Softmax
pyvqnet.nn.Softplus
pyvqnet.nn.Softsign
pyvqnet.nn.Sigmoid
pyvqnet.nn.HardSigmoid

# norm
pyvqnet.nn.BatchNorm1d
pyvqnet.nn.BatchNorm2d
pyvqnet.nn.LayerNorm1d
pyvqnet.nn.LayerNorm2d
pyvqnet.nn.LayerNormNd
pyvqnet.nn.Spectral_Norm

# loss
pyvqnet.nn.BinaryCrossEntropy
pyvqnet.nn.CategoricalCrossEntropy
pyvqnet.nn.CrossEntropyLoss
pyvqnet.nn.SoftmaxCrossEntropy
pyvqnet.nn.MeanSquaredError
pyvqnet.nn.NLL_Loss
pyvqnet.nn.loss.fidelityLoss
```

### QNN

```python
# basic layer
pyvqnet.qnn.VQCLayer
pyvqnet.qnn.VQC_wrapper
pyvqnet.qnn.QuantumLayer
pyvqnet.qnn.QuantumLayerV2
pyvqnet.qnn.QuantumLayerMultiProcess
pyvqnet.qnn.QuantumVariableLayer
pyvqnet.qnn.NoiseQuantumLayer
pyvqnet.qnn.quantumlayer.calculate_gain
pyvqnet.qnn.quantumlayer.delayed
pyvqnet.qnn.quantumlayer.updatevar
pyvqnet.qnn.quantumlayer.ones
pyvqnet.qnn.quantumlayer.zeros
pyvqnet.qnn.quantumlayer.normal
pyvqnet.qnn.quantumlayer.uniform
pyvqnet.qnn.quantumlayer.he_normal
pyvqnet.qnn.quantumlayer.he_uniform
pyvqnet.qnn.quantumlayer.xavier_normal
pyvqnet.qnn.quantumlayer.xavier_uniform
pyvqnet.qnn.quantumlayer.quantum_uniform

# circuit templates
pyvqnet.qnn.template.RandomTemplate
pyvqnet.qnn.BasicEmbeddingCircuit
pyvqnet.qnn.AmplitudeEmbeddingCircuit
pyvqnet.qnn.AngleEmbeddingCircuit
pyvqnet.qnn.IQPEmbeddingCircuits
pyvqnet.qnn.BasicEntanglerTemplate
pyvqnet.qnn.StronglyEntanglingTemplate
pyvqnet.qnn.RotCircuit
pyvqnet.qnn.CRotCircuit
pyvqnet.qnn.CSWAPcircuit
pyvqnet.qnn.HardwareEfficientAnsatz

# measure
pyvqnet.qnn.measure.expval
pyvqnet.qnn.measure.QuantumMeasure
pyvqnet.qnn.measure.ProbsMeasure
pyvqnet.qnn.measure.DensityMatrixFromQstate
pyvqnet.qnn.measure.Mutal_Info
pyvqnet.qnn.measure.VN_Entropy

# quantum variational circuit
pyvqnet.qnn.qvc.Qvc
pyvqnet.qnn.qvc.qvc_model.Anchor
pyvqnet.qnn.qvc.qvc_model.D
pyvqnet.qnn.qvc.qvc_model.DEFAULT_SCALE
pyvqnet.qnn.qvc.qvc_model.DefaultStyle
pyvqnet.qnn.qvc.qvc_model.FRAC_MESH
pyvqnet.qnn.qvc.qvc_model.HAS_MATPLOTLIB
pyvqnet.qnn.qvc.qvc_model.HIG
pyvqnet.qnn.qvc.qvc_model.MatplotlibDrawer
pyvqnet.qnn.qvc.qvc_model.N
pyvqnet.qnn.qvc.qvc_model.PORDER_GATE
pyvqnet.qnn.qvc.qvc_model.PORDER_GRAY
pyvqnet.qnn.qvc.qvc_model.PORDER_LINE
pyvqnet.qnn.qvc.qvc_model.PORDER_REGLINE
pyvqnet.qnn.qvc.qvc_model.PORDER_SUBP
pyvqnet.qnn.qvc.qvc_model.PORDER_TEXT
pyvqnet.qnn.qvc.qvc_model.Parameter
pyvqnet.qnn.qvc.qvc_model.ParameterExpression
pyvqnet.qnn.qvc.qvc_model.QIndexError
pyvqnet.qnn.qvc.qvc_model.QUserConfigError
pyvqnet.qnn.qvc.qvc_model.Qvc
pyvqnet.qnn.qvc.qvc_model.WID
pyvqnet.qnn.qvc.qvc_model.build_circult
pyvqnet.qnn.qvc.qvc_model.draw_circuit_pic
pyvqnet.qnn.qvc.qvc_model.get_backend
pyvqnet.qnn.qvc.qvc_model.get_cnot
pyvqnet.qnn.qvc.qvc_model.my_cbit
pyvqnet.qnn.qvc.qvc_model.my_qubit
pyvqnet.qnn.qvc.qvc_model.qvc_circuits_with_noise

# quantum deep reinforcement learning
pyvqnet.qnn.qdrl.vmodel
pyvqnet.qnn.qdrl.vqnet_model.build_circult
pyvqnet.qnn.qdrl.vqnet_model.get_grad
pyvqnet.qnn.qdrl.vqnet_model.qdrl_circuit
pyvqnet.qnn.qdrl.vqnet_model.vmodel

# linear
pyvqnet.qnn.qlinear.QLinear
pyvqnet.qnn.qlinear.qlinear.qlinear_circuit

# conv
pyvqnet.qnn.qcnn.QConv
pyvqnet.qnn.qcnn.functions_conv.col2im_array
pyvqnet.qnn.qcnn.functions_conv.im2col_array
pyvqnet.qnn.qcnn.functions_conv.deconv2d
pyvqnet.qnn.qcnn.functions_conv.get_conv_outsize
pyvqnet.qnn.qcnn.functions_conv.pair
pyvqnet.qnn.qcnn.functions_conv.unwrap_padding
pyvqnet.qnn.qcnn.qconv.CZ_layer
pyvqnet.qnn.qcnn.qconv.RY_layer
pyvqnet.qnn.qcnn.qconv.latent_layer
pyvqnet.qnn.qcnn.qconv.encode_cir
pyvqnet.qnn.qcnn.qconv.entangle_cir
pyvqnet.qnn.qcnn.qconv.param_cir
pyvqnet.qnn.qcnn.qconv.qcnn_circuit

# what's this ?
pyvqnet.qnn.pqc.PQCLayer

# quantum autoencoder
pyvqnet.qnn.qae.QAElayer

# quantum svm
pyvqnet.qnn.svm.QuantumKernel_VQNet
pyvqnet.qnn.svm.gen_vqc_qsvm_data
pyvqnet.qnn.svm.vqc_qsvm
pyvqnet.qnn.svm.vqc_svm.SPSA

# adapters to other framework
pyvqnet.qnn.utils.Compatiblelayer
pyvqnet.qnn.utils.CirqLayer
pyvqnet.qnn.utils.QiskitLayer
pyvqnet.qnn.utils.QiskitLayerV2
```

### Optim

```python
# classical gradient-based
pyvqnet.optim.SGD
pyvqnet.optim.Adagrad
pyvqnet.optim.Adadelta
pyvqnet.optim.RMSProp
pyvqnet.optim.Adam
pyvqnet.optim.Adamax

# quantum-specified
pyvqnet.optim.Rotosolve
```

### Misc & Utils

```python
# version
pyvqnet.VERSION

# data
pyvqnet.data.data_generator

# metrics
pyvqnet.utils.metrics.MAE
pyvqnet.utils.metrics.MAPE
pyvqnet.utils.metrics.SMAPE
pyvqnet.utils.metrics.MSE
pyvqnet.utils.metrics.RMSE
pyvqnet.utils.metrics.R_Square
pyvqnet.utils.metrics.auc_calculate
pyvqnet.utils.metrics.auc_calculate_old
pyvqnet.utils.metrics.precision_recall_f1_2_score
pyvqnet.utils.metrics.precision_recall_f1_N_score
pyvqnet.utils.metrics.precision_recall_f1_Multi_score

# save/load
pyvqnet.utils.storage.load_parameters
pyvqnet.utils.storage.save_parameters

# utils
pyvqnet.utils.utils.accuracy_score
pyvqnet.utils.utils.batch_iterator
pyvqnet.utils.utils.default_noise_config
pyvqnet.utils.utils.dilate_input
pyvqnet.utils.utils.get_deconv_outsize
pyvqnet.utils.utils.make_diagonal
pyvqnet.utils.utils.make_padding
pyvqnet.utils.utils.normalize
pyvqnet.utils.utils.transpose_kernel
# pyvqnet.qnn.qcnn.functions_conv
pyvqnet.utils.utils.pair
pyvqnet.utils.utils.get_conv_outsize
pyvqnet.utils.utils.unwrap_padding
# pyvqnet.qnn.utils.compatible_layer
pyvqnet.utils.utils.bind_cirq_symbol
pyvqnet.utils.utils.get_circuit_symbols
pyvqnet.utils.utils.merge_cirq_paramsolver
pyvqnet.utils.utils.validate_compatible_grad
pyvqnet.utils.utils.validate_compatible_input
pyvqnet.utils.utils.validate_compatible_output
```

----
by Armit
2023年4月14日
