#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/18

# https://pennylane.ai/qml/demos/tutorial_variational_classifier/

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit(weights, x):
    state_preparation(x)
    for layer_weights in weights:
        layer(layer_weights)
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

# ↑↑↑ common to all qnn clf, specific to the problem ↓↓↓

def state_preparation(x):
    qml.BasisState(x, wires=[0, 1, 2, 3])

def layer(layer_weights):
    for wire in range(4):
        qml.Rot(*layer_weights[wire], wires=wire)
    for wires in ([0, 1], [1, 2], [2, 3], [3, 0]):
        qml.CNOT(wires)

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)


if __name__ == '__main__':
    data = np.loadtxt("parity_train.txt", dtype=int)
    X = np.array(data[:, :-1])
    Y = np.array(data[:, -1])
    Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}
    for x, y in zip(X, Y):
        print(f"x = {x}, y = {y}")

    np.random.seed(0)
    num_qubits = 4
    num_layers = 2
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)
    print("Weights:", weights_init)
    print("Bias: ", bias_init)

    opt = NesterovMomentumOptimizer(0.5)
    batch_size = 5
    weights = weights_init
    bias = bias_init

    for it in range(100):
        # Update the weights by one optimizer step, using only a limited batch of data
        batch_index = np.random.randint(0, len(X), (batch_size,))
        X_batch = X[batch_index]
        Y_batch = Y[batch_index]
        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)

        # Compute accuracy
        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]

        current_cost = cost(weights, bias, X, Y)
        acc = accuracy(Y, predictions)

        print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")

    data = np.loadtxt("parity_test.txt", dtype=int)
    X_test = np.array(data[:, :-1])
    Y_test = np.array(data[:, -1])
    Y_test = Y_test * 2 - 1  # shift label from {0, 1} to {-1, 1}

    predictions_test = [np.sign(variational_classifier(weights, bias, x)) for x in X_test]

    for x, y, p in zip(X_test, Y_test, predictions_test):
        print(f"x = {x}, y = {y}, pred={p}")

    acc_test = accuracy(Y_test, predictions_test)
    print("Accuracy on unseen data:", acc_test)