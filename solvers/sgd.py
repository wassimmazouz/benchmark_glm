from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def gradient_multilogreg(X, y, w):
    z = softmax(np.matmul(X, w))
    return (np.dot(X.T, (z - y)))/len(X)


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
        step = 1 / L

        X, y = self.X, self.y

        w = np.zeros((n_features, y.shape[1]))
        for _ in range(n_iter):
            prm = np.random.permutation(n_samples)  # Shuffle the data for SGD
            for i in prm:
                xi = X[i:i+1]
                yi = y[i:i+1]
                gradient = gradient_multilogreg(xi, yi, w)
                w -= step * gradient

        self.w = w

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        return np.linalg.norm(self.X, ord=2)**2 / self.X.shape[0]
