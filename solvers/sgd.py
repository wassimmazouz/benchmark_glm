from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from scipy import sparse


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return (exp_z.T / np.sum(exp_z, axis=1)).T


def gradient_multilogreg(X, y, w):
    z = softmax(X @ w)
    return X.T @ (z - y)


def objective_function_logreg(X, y, beta):
    y_X_beta = y * (X @ beta)
    return np.log1p(np.exp(-y_X_beta)).sum()


def objective_function_linreg(X, y, beta):
    diff = y - X @ beta
    return 5 * diff @ diff


def objective_function_multilogreg(X, y, w):
    w = w.reshape((X.shape[1], y.shape[1]))
    z = softmax(X @ w)
    return -np.sum(y * np.log(z))


def gradient_logreg(X, y, beta):
    y_X_beta = y * (X @ beta)
    return -(X.T @ (y * sigmoid(y_X_beta)))


def gradient_linreg(X, y, beta):
    return X.T @ (X @ beta - y)


class Solver(BaseSolver):
    name = 'SGD'  # stochastic gradient descent
    parameters = {'step_scale': [0.1, 0.2, 0.5, 1, 1.2]}

    def set_objective(self, X, y, model):
        self.X, self.y, self.model = X, y, model

    def run(self, n_iter):

        n_samples, n_features = self.X.shape

        L = self.compute_lipschitz_constant()
        step = self.step_scale / L

        X, y = self.X, self.y

        if self.model == 'multilogreg':
            w = np.zeros((n_features, y.shape[1]))
            gradient = gradient_multilogreg
        if self.model == 'linreg':
            w = np.zeros(n_features)
            gradient = gradient_linreg
        if self.model == 'logreg':
            w = np.zeros(n_features)
            gradient = gradient_logreg

        for _ in range(n_iter):
            prm = np.random.permutation(n_samples)  # Shuffle the data for SGD
            for i in prm:
                xi = X[i:i+1]
                yi = y[i:i+1]
                grad = gradient(xi, yi, w)
                w -= step * grad

        self.w = w

    def get_result(self):
        return dict(beta=self.w)

    def compute_lipschitz_constant(self):
        if self.model == 'multilogreg':
            return np.linalg.norm(self.X, ord=2)**2

        if self.model == 'logreg':
            if not sparse.issparse(self.X):
                L = np.linalg.norm(self.X, ord=2) ** 2 / 4
            else:
                L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2 / 4
            return L

        if self.model == 'linreg':
            return np.linalg.norm(self.X, ord=2) ** 2
