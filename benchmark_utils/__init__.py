import numpy as np


def gradient_linreg(X, y, beta):
    return X.T @ (X @ beta - y)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_logreg(X, y, beta):
    m = len(y)
    z = np.dot(X, beta)
    h = sigmoid(z)
    gradient = np.dot(X.T, (h - y)) / m
    return gradient


def objective_function_logreg(X, y, beta):
    y_X_beta = y * X.dot(beta.flatten())
    return np.log1p(np.exp(-y_X_beta)).sum()


def objective_function_linreg(X, y, beta):
    diff = y - X @ beta
    return 5 * diff @ diff


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    soft = (exp_z.T / np.sum(exp_z, axis=1)).T
    return soft


def gradient_multilogreg(X, y, w):
    z = softmax(np.matmul(X, w))
    result = -(np.dot(X.T, (y - z)))/len(X)
    return result


def objective_function_multilogreg(X, y, w):
    z = softmax(np.matmul(X, w))
    result = -(np.sum(y * np.log(z)))/len(X)
    return result
