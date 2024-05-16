from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse
    from benchmark_utils import softmax


class Solver(BaseSolver):
    name = 'SGD'  # gradient descent

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

        row = X.shape[0]  # Initialize the row for "stochastic"

        X = np.insert(X, 0, 1, axis=1)  # Add constant vector

        w = np.zeros(n_features)
        for i in range(n_iter):

            prm = np.random.permutation(row)  # Initialize the "stochastic"
            for i in prm:
                soft = softmax(w @ X[i])  # Softmax
                gradient = np.outer(soft - y[i], X[i])  # Stochastic gradient descent
                w -= - L * gradient  # Update weight value

        self.w = w

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2 / 4
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / 4
        return L
