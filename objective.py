from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.obj_helper import objective_function_linreg, objective_function_logreg


class Objective(BaseObjective):
    min_benchopt_version = "1.5"
    name = "GLM"

    parameters = {
        'model': ['linreg', 'logreg']
    }

    def __init__(self, model='linreg'):
        self.model = model

    def set_data(self, X, y, dataset_model):
        if self.model == 'logreg' and self.model in dataset_model:
            if set(y) != set([-1, 1]):
                raise ValueError(
                    f"y must contain only -1 or 1 as values. Got {set(y)}"
                )

        self.X, self.y, self.dataset_model = X, y, dataset_model

    def get_one_result(self):
        n_features = self.X.shape[1]
        return dict(beta=np.zeros(n_features))

    def evaluate_result(self, beta):
        X, y, = self.X, self.y
        if self.model == 'logreg':

            return objective_function_logreg(X, y, beta)

        if self.model == 'linreg':
            return objective_function_linreg(X, y, beta)

    def get_objective(self):
        return dict(X=self.X, y=self.y, model=self.model, dataset_model=self.dataset_model)
