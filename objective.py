from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils import sigmoid


class Objective(BaseObjective):
    min_benchopt_version = "1.5"
    name = "GLM"

    parameters = {
        'whiten_y': [False, True],
        'fit_intercept': [False],
        'model': ['linreg', 'logreg']
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
            self.X, self.y = X, y

        if self.model == 'linreg':
            self.X, self.y = X, y

            if self.whiten_y:
                y -= y.mean(axis=0)

    def get_one_result(self):
        n_features = self.X.shape[1]
        if self.fit_intercept:
            n_features += 1
        return dict(beta=np.zeros(n_features))

    def evaluate_result(self, beta, ):
        if self.model == 'logreg':
            X, y = self.X, self.y
            h = sigmoid(X @ beta)

            return -(y.T @ np.log(h) + (1 - y).T @ np.log(1 - h)) / len(y)

        if self.model == 'linreg':
            diff = self.y - self.X @ beta

            return dict(
                value=.5 * diff @ diff,
            )

    def get_objective(self):
        return dict(X=self.X, y=self.y, model=self.model)
