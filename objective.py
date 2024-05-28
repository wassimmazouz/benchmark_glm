from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    min_benchopt_version = "1.5"
    name = "GLM"

    parameters = {
        'model': ['linreg', 'logreg']
    }

    def __init__(self, model='linreg'):
        self.model = model

    def set_data(self, X, y):
        if self.model == 'logreg':
            if set(y) != set([-1, 1]):
                raise ValueError(
                    f"y must contain only -1 or 1 as values. Got {set(y)}"
                )

        self.X, self.y = X, y

    def get_one_result(self):
        n_features = self.X.shape[1]
        return dict(beta=np.zeros(n_features))

    def evaluate_result(self, beta):
        X, y, = self.X, self.y
        if self.model == 'logreg':
            y_X_beta = y * (X @ beta.flatten())

            return np.log1p(np.exp(-y_X_beta)).sum()

        if self.model == 'linreg':
            diff = y - X @ beta

            return dict(
                value=.5 * diff @ diff,
            )

    def get_objective(self):
        return dict(X=self.X, y=self.y, model=self.model)
