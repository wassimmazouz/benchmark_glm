from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from benchmark_utils.grad_helper import gradient_linreg, gradient_logreg


class Solver(BaseSolver):
    name = 'Python-GD'  # gradient descent

    def set_objective(self, X, y, model, dataset_model):
        self.X, self.y, self.model, self.dataset_model = X, y, model, dataset_model

    def skip(self, X, y, model, dataset_model):
        if model not in dataset_model or model not in ['linreg', 'logreg']:
            return True, f"{model} not suitable for this dataset"

        return False, None

    def gradient(self, X, y, w):
        if self.model == 'logreg':
            return gradient_logreg(X, y, w)
        if self.model == 'linreg':
            return gradient_linreg(X, y, w)

    def run(self, n_iter):
        n_samples, n_features = self.X.shape

        L = self.compute_lipschitz_constant()
        step = 1. / L

        X, y = self.X, self.y

        w = np.zeros(n_features)
        for _ in range(n_iter):
            grad = self.gradient(X, y, w)
            w -= step * grad

        self.w = w

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        if self.model == 'logreg':
            if not sparse.issparse(self.X):
                L = np.linalg.norm(self.X, ord=2) ** 2 / 4
            else:
                L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / 4
            return L
        if self.model == 'linreg':
            return np.linalg.norm(self.X, ord=2) ** 2
