from benchopt import BaseObjective, safe_import_context
from solvers import

with safe_import_context() as import_ctx:
    import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return (exp_z.T / np.sum(exp_z, axis=1)).T


def objective_function_multilogreg(X, y, w):
    z = softmax(np.matmul(X, w))
    return -(np.sum(y * np.log(z)))/len(X)


class Objective(BaseObjective):
    min_benchopt_version = "1.5"
    name = "GLM"

    parameters = {
        'whiten_y': [False, True],
        'fit_intercept': [False],
        'model': ['linreg', 'logreg', 'multilogreg']
    }

    def __init__(self, fit_intercept=False, whiten_y=False, model='linreg'):
        self.fit_intercept = fit_intercept
        self.whiten_y = whiten_y
        self.model = model

    def set_data(self, X, y):
        if self.model == 'logreg':
            if set(y) != set([-1, 1]):
                raise ValueError(
                    f"y must contain only -1 or 1 as values. Got {set(y)}"
                )

        if self.model == 'linreg' and self.whiten_y:
            y -= y.mean(axis=0)

        self.X, self.y = X, y

    def get_one_result(self):
        n_features = self.X.shape[1]
        if self.fit_intercept:
            n_features += 1
        return dict(beta=np.zeros(n_features))

    def evaluate_result(self, beta):
        X, y, = self.X, self.y
        if self.model == 'logreg':
            y_X_beta = y * (X @ beta.flatten())

            return np.log1p(np.exp(-y_X_beta)).mean()

        if self.model == 'linreg':
            diff = y - X @ beta

            return dict(
                value=.5 * diff @ diff,
            )

        if self.model == 'multilogreg':
            return objective_function_multilogreg(X, y, beta)

    def get_objective(self):
        return dict(X=self.X, y=self.y, model=self.model)
