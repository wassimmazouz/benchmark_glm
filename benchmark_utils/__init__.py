import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_lr(X, y, beta):
    return X.T @ (X @ beta - y)


def objective_function(X, y, beta):
    h = sigmoid(X @ beta)
    return -(y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) / len(y)
