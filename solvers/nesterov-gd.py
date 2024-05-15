from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np

    # import your reusable functions here
    from benchmark_utils import gradient_lr


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'NESTEROV-GD'

    # added momentum parameter for the nesterov acceleration
    parameters = {
        'scale_step': [0.5, 0.99, 1.2], 'momentum_parameter': [0, 0.2, 0.5, 1]
    }

    # List of packages needed to run the solver. See the corresponding
    # section in objective.py
    requirements = []

    def set_objective(self, X, y):
        self.X, self.y = X, y

    def run(self, n_iter):

        L = np.linalg.norm(self.X, ord=2) ** 2
        step_size = self.scale_step / L
        beta = np.zeros(self.X.shape[1])
        momentum = 0
        for _ in range(n_iter):
            momentum = self.momentum_parameter * momentum - step_size * \
                gradient_lr(self.X, self.y, beta + self.momentum_parameter * momentum)
            beta += momentum
        self.beta = beta

    def get_result(self):
        return dict(beta=self.beta)
