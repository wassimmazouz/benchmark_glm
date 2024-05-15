# `benchmark_utils` is a module in which you can define code to reuse in
# the benchmark objective, datasets, and solvers. The folder should have the
# name `benchmark_utils`, and code defined inside will be importable using
# the usual import syntax
import numpy as np


def gradient_lr(X, y, beta):
    return X.T @ (X @ beta - y)


def objective_function(X, y, beta):
    beta = beta.flatten().astype(np.float64)
    y_X_beta = y * (X @ beta)
    return np.log(1 + np.exp(-y_X_beta)).sum()
