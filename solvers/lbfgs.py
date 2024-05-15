from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np
    from scipy.optimize import minimize

    # Here, useful functions are imported from the benchmark_utils file
    from benchmark_utils import objective_function


#
class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'L-BFGS-B'

    parameters = {}

    requirements = []

    def set_objective(self, X, y):

        self.X, self.y = X, y

    def run(self, n_iter):

        def fun(beta): return objective_function(self.X, self.y, beta)
        beta_0 = np.zeros(self.X.shape[1])
        result = minimize(fun, beta_0, method='L-BFGS-B', options={'maxiter': n_iter})
        self.beta = result.x

    def get_result(self):
        return dict(beta=self.beta)
