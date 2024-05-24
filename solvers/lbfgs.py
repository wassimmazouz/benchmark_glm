from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import minimize


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return (exp_z.T / np.sum(exp_z, axis=1)).T


def objective_function_logreg(X, y, beta):
    y_X_beta = y * (X @ beta.flatten())
    return np.log1p(np.exp(-y_X_beta)).sum()


def objective_function_linreg(X, y, beta):
    diff = y - X @ beta
    return 5 * diff @ diff


def objective_function_multilogreg(X, y, w):
    w = w.reshape((X.shape[1], y.shape[1]))
    z = softmax(X @ w)
    return -np.sum(y * np.log(z))


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

        if self.model == 'multilogreg':
            def fun(beta): return objective_function_multilogreg(
                self.X, self.y, beta)

        beta_0 = np.zeros(self.X.shape[1])

        if self.model == 'multilogreg':
            beta_0 = np.zeros((self.X.shape[1], self.y.shape[1])).flatten()

        result = minimize(fun, beta_0, method='L-BFGS-B', options={'maxiter': n_iter})
        self.beta = result.x
        if self.model == 'multilogreg':
            self.beta = result.x.reshape((self.X.shape[1], self.y.shape[1]))

    def get_result(self):
        return dict(beta=self.beta)
