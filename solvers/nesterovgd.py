from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from benchmark_utils.grad_helper import gradient_linreg, gradient_logreg


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'NESTEROV-GD'

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def gradient(self, X, y, w):
        if self.model == 'logreg':
            return gradient_logreg(X, y, w)
        if self.model == 'linreg':
            return gradient_linreg(X, y, w)

    def set_objective(self, X, y, model):
        self.X, self.y, self.model = X, y, model

    def run(self, n_iter):

        step_size = 1. / self.compute_lipschitz_constant()
        beta = np.zeros(self.X.shape[1])
        alpha = beta.copy()
        t = 1

        for _ in range(n_iter):
            t_next = (1 + np.sqrt(1+4*t**2)) / 2
            beta_next = alpha - step_size * self.gradient(self.X, self.y, alpha)
            alpha = beta_next + ((t - 1) / t_next) * (beta_next - beta)
            t, beta = t_next, beta_next

        self.beta = beta

    def get_result(self):
        return dict(beta=self.beta)

    def compute_lipschitz_constant(self):
        if self.model == 'logreg':
            if not sparse.issparse(self.X):
                L = np.linalg.norm(self.X, ord=2) ** 2 / 4
            else:
                L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / 4
            return L
        if self.model == 'linreg':
            return np.linalg.norm(self.X, ord=2) ** 2
