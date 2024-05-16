import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_ols(X, y, beta):
    return X.T @ (X @ beta - y)


def objective_function_logreg(X, y, beta):
    h = sigmoid(X @ beta)
    return -(y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) / len(y)


def objective_function_linreg(X, y, beta):
    diff = y - X @ beta
    return 5 * diff @ diff
