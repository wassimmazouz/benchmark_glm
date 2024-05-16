from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


class Solver(BaseSolver):
    name = 'Python-GD'  # gradient descent

    def set_objective(self, X, y):
        self.X, self.y = X, y

    def run(self, n_iter):
        n_samples, n_features = self.X.shape

        L = self.compute_lipschitz_constant()
        step = 1. / L

        X, y = self.X, self.y

        w = np.zeros(n_features)
        for i in range(n_iter):
            ywTx = y * (X @ w)
            temp = 1. / (1. + np.exp(ywTx))
            grad = -(X.T @ (y * temp))
            w -= step * grad

        self.w = w

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2 / 4
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / 4
        return L
