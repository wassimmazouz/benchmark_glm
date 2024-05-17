from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from benchmark_utils import gradient_multilogreg


class Solver(BaseSolver):
    name = 'SGD'  # stochastic gradient descent

    def set_objective(self, X, y, model):
        self.X, self.y, self.model = X, y, model

        if model != 'multilogreg':
            print("Please choose the 'multilogreg' to use SGD solver")
            # Stop execution by raising an exception
            raise ValueError("Wrong model for solver")

    def run(self, n_iter):

        n_samples, n_features = self.X.shape

        L = self.compute_lipschitz_constant()
        step = 1. / L

        X, y = self.X, self.y

        w = np.zeros((n_features, y.shape[1]))
        for _ in range(n_iter):

            prm = np.random.permutation(n_samples)  # Initialize the "stochastic"
            for i in prm:
                gradient = gradient_multilogreg(X, y, w)
                w -= step * gradient  # Update weight value

        self.w = w

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        X = self.X
        XT_X = np.dot(X.T, X)
        largest_eigenvalue = np.linalg.norm(XT_X, ord=2)
        return largest_eigenvalue / 4
