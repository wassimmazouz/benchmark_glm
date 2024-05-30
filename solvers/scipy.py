from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from scipy.sparse.linalg import cgs, tfqmr


class Solver(BaseSolver):
    name = "scipy"

    requirements = ["scipy>=1.8"]
    install_cmd = "conda"
    parameters = {"solver": ["cgs", "tfqmr"]}

    def set_objective(self, X, y, model, dataset_model):
        self.X, self.y, self.model, self.dataset_model = X, y, model, dataset_model

    def skip(self, X, y, model, dataset_model):
        if model not in dataset_model or model not in ['linreg']:
            return True, "model not suitable for this dataset"

        return False, None

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
