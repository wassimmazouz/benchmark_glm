from scipy.special import expit
import numpy as np


def gradient_linreg(X, y, beta):
    return X.T @ (X @ beta - y)


def gradient_logreg(X, y, beta):
    y_X_beta = y * (X @ beta)
    return -(X.T @ (y * expit(-y_X_beta)))
