from benchopt import BaseSolver, safe_import_context


with safe_import_context() as ctx:
    from scipy.sparse.linalg import cgs, tfqmr


class Solver(BaseSolver):
    name = "scipy"

    requirements = ["scipy>=1.8"]
    install_cmd = "conda"
    parameters = {"solver": ["cgs", "tfqmr"]}

    def set_objective(self, X, y, model, fit_intercept=False):
        if model != 'linreg':
            print("Please choose the 'linreg' to use scipy solver")
            # Stop execution by raising an exception
            raise ValueError("Wrong model for solver")

        self.X, self.y = X, y
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        X, y = self.X, self.y
        if self.solver == "cgs":
            algo = cgs
        elif self.solver == "tfqmr":
            algo = tfqmr
        else:
            raise ValueError(f"Unknown solver {self.solver}")

        x, _ = algo(X.T @ X, X.T @ y, maxiter=n_iter, tol=1e-12)
        self.w = x

    def get_result(self):
        return dict(beta=self.w)
