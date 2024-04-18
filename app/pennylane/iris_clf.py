#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/18 

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt

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

def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))
    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

def state_preparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)

if 'test state_preparation':
    x = np.array([0.53896774, 0.79503606, 0.27826503, 0.0], requires_grad=False)
    ang = get_angles(x)

    @qml.qnode(dev)
    def test(angles):
        state_preparation(angles)
        return qml.state()

    state = test(ang)
    print("x               : ", np.round(x, 6))
    print("angles          : ", np.round(ang, 6))
    print("amplitude vector: ", np.round(np.real(state), 6))

def layer(layer_weights):
    for wire in range(2):
        qml.Rot(*layer_weights[wire], wires=wire)
    qml.CNOT(wires=[0, 1])

def cost(weights, bias, X, Y):
    # Transpose the batch of input data in order to make the indexing
    # in state_preparation work
    predictions = variational_classifier(weights, bias, X.T)
    return square_loss(Y, predictions)


if __name__ == '__main__':
    data = np.loadtxt("iris_classes1and2_scaled.txt")
    X = data[:, 0:2]
    Y = data[:, -1]
    print(f"First X sample (original)  : {X[0]}")
    # pad the vectors to size 2^2=4 with constant values
    padding = np.ones((len(X), 2)) * 0.1
    X_pad = np.c_[X, padding]
    print(f"First X sample (padded)    : {X_pad[0]}")
    # normalize each input
    normalization = np.sqrt(np.sum(X_pad**2, -1))
    X_norm = (X_pad.T / normalization).T
    print(f"First X sample (normalized): {X_norm[0]}")
    # the angles for state preparation are the features
    features = np.array([get_angles(x) for x in X_norm], requires_grad=False)
    print(f"First features sample      : {features[0]}")

    if not 'plot data':
        plt.figure()
        plt.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], c="b", marker="o", ec="k")
        plt.scatter(X[:, 0][Y == -1], X[:, 1][Y == -1], c="r", marker="o", ec="k")
        plt.title("Original data")
        plt.show()

        plt.figure()
        dim1 = 0
        dim2 = 1
        plt.scatter(X_norm[:, dim1][Y == 1], X_norm[:, dim2][Y == 1], c="b", marker="o", ec="k")
        plt.scatter(X_norm[:, dim1][Y == -1], X_norm[:, dim2][Y == -1], c="r", marker="o", ec="k")
        plt.title(f"Padded and normalised data (dims {dim1} and {dim2})")
        plt.show()

        plt.figure()
        dim1 = 0
        dim2 = 3
        plt.scatter(features[:, dim1][Y == 1], features[:, dim2][Y == 1], c="b", marker="o", ec="k")
        plt.scatter(features[:, dim1][Y == -1], features[:, dim2][Y == -1], c="r", marker="o", ec="k")
        plt.title(f"Feature vectors (dims {dim1} and {dim2})")
        plt.show()

    np.random.seed(0)
    num_qubits = 4
    num_layers = 2
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)
    print("Weights:", weights_init)
    print("Bias: ", bias_init)

    num_data = len(Y)
    num_train = int(0.75 * num_data)
    index = np.random.permutation(range(num_data))
    feats_train = features[index[:num_train]]
    Y_train = Y[index[:num_train]]
    feats_val = features[index[num_train:]]
    Y_val = Y[index[num_train:]]

    # We need these later for plotting
    X_train = X[index[:num_train]]
    X_val = X[index[num_train:]]

    opt = NesterovMomentumOptimizer(0.01)
    batch_size = 5
    weights = weights_init
    bias = bias_init

    # train the variational classifier
    for it in range(100):
        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, num_train, (batch_size,))
        feats_train_batch = feats_train[batch_index]
        Y_train_batch = Y_train[batch_index]
        weights, bias, _, _ = opt.step(cost, weights, bias, feats_train_batch, Y_train_batch)

        # Compute predictions on train and validation set
        predictions_train = np.sign(variational_classifier(weights, bias, feats_train.T))
        predictions_val = np.sign(variational_classifier(weights, bias, feats_val.T))

        # Compute accuracy on train and validation set
        acc_train = accuracy(Y_train, predictions_train)
        acc_val = accuracy(Y_val, predictions_val)

        if (it + 1) % 2 == 0:
            _cost = cost(weights, bias, features, Y)
            print(
                f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
                f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
            )

        if (it + 1) % 10 == 0:
            plt.figure()
            cm = plt.cm.RdBu

            # make data for decision regions
            xx, yy = np.meshgrid(np.linspace(0.0, 1.5, 30), np.linspace(0.0, 1.5, 30))
            X_grid = [np.array([x, y]) for x, y in zip(xx.flatten(), yy.flatten())]

            # preprocess grid points like data inputs above
            padding = 0.1 * np.ones((len(X_grid), 2))
            X_grid = np.c_[X_grid, padding]  # pad each input
            normalization = np.sqrt(np.sum(X_grid**2, -1))
            X_grid = (X_grid.T / normalization).T  # normalize each input
            features_grid = np.array([get_angles(x) for x in X_grid])  # angles are new features
            predictions_grid = variational_classifier(weights, bias, features_grid.T)
            Z = np.reshape(predictions_grid, xx.shape)

            # plot decision regions
            levels = np.arange(-1, 1.1, 0.1)
            cnt = plt.contourf(xx, yy, Z, levels=levels, cmap=cm, alpha=0.8, extend="both")
            plt.contour(xx, yy, Z, levels=[0.0], colors=("black",), linestyles=("--",), linewidths=(0.8,))
            plt.colorbar(cnt, ticks=[-1, 0, 1])

            # plot data
            for color, label in zip(["b", "r"], [1, -1]):
                plot_x = X_train[:, 0][Y_train == label]
                plot_y = X_train[:, 1][Y_train == label]
                plt.scatter(plot_x, plot_y, c=color, marker="o", ec="k", label=f"class {label} train")
                plot_x = (X_val[:, 0][Y_val == label],)
                plot_y = (X_val[:, 1][Y_val == label],)
                plt.scatter(plot_x, plot_y, c=color, marker="^", ec="k", label=f"class {label} validation")

            plt.legend()
            plt.show()
