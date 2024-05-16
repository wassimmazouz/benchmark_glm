import numpy as np


def gradient_linreg(X, y, beta):
    return X.T @ (X @ beta - y)


def gradient_logreg(X, y, w):
    ywTx = y * (X @ w)
    temp = 1. / (1. + np.exp(ywTx))
    grad = -(X.T @ (y * temp))
    return grad


def objective_function_logreg(X, y, beta):
    y_X_beta = y * X.dot(beta.flatten())
    return np.log1p(np.exp(-y_X_beta)).sum()


def objective_function_linreg(X, y, beta):
    diff = y - X @ beta
    return 5 * diff @ diff


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def objective_function_multilogreg(X, y, w):
    L = 0
    for i in range(X.shape[0]):
        soft = self.softmax(w @ X[i])
        L -= np.dot(y[i], np.log(soft))
    return L
