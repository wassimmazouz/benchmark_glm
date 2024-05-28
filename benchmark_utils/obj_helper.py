import numpy as np


def objective_function_linreg(X, y, beta):
    diff = y - X @ beta
    return 5 * diff @ diff


def objective_function_logreg(X, y, beta):
    y_X_beta = y * (X @ beta)
    return np.log1p(np.exp(-y_X_beta)).sum()
