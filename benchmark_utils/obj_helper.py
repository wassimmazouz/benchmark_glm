import numpy as np
import torch
from scipy.special import expit


def objective_function_linreg(X, y, beta):
    diff = y - X @ beta
    return 0.5 * diff @ diff


def objective_function_logreg(X, y, beta):
    y_X_beta = y * (X @ beta)

    if isinstance(X, torch.Tensor):
        return -torch.log(torch.sigmoid(y_X_beta)).sum()
    else:
        return -np.log(expit(y_X_beta)).sum()
