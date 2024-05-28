from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import minimize


def objective_function_logreg(X, y, beta):
    y_X_beta = y * (X @ beta)
    return np.log1p(np.exp(-y_X_beta)).sum()


def objective_function_linreg(X, y, beta):
    diff = y - X @ beta
    return 5 * diff @ diff


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'L-BFGS-B'

    parameters = {}

    requirements = []

    def set_objective(self, X, y, model):
        self.X, self.y, self.model = X, y, model

    def run(self, n_iter):
        if self.model == 'logreg':
            def fun(beta): return objective_function_logreg(
                self.X, self.y, beta)

        if self.model == 'linreg':
            def fun(beta): return objective_function_linreg(
                self.X, self.y, beta)

        beta_0 = np.zeros(self.X.shape[1])

        result = minimize(fun, beta_0, method='L-BFGS-B', options={'maxiter': n_iter})
        self.beta = result.x

    def get_result(self):
        return dict(beta=self.beta)
