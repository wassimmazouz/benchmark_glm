from scipy.special import expit
import numpy as np


def gradient_linreg(X, y, beta):
    return X.T @ (X @ beta - y)


def gradient_logreg(X, y, beta):
    # return X.T @ (expit(X @ beta) - y)
    y_X_beta = y * (X @ beta)
    # temp = 1. / (1. + np.exp(ywTx))
    # grad = -(X.T @ (y * temp))
    return -(X.T @ (y * expit(-y_X_beta)))
