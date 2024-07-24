import warnings


from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import PoissonRegressor
    from sklearn.linear_model import SGDClassifier
    from scipy.optimize.linesearch import LineSearchWarning
    import numpy as np


class Solver(BaseSolver):
    name = 'sklearn'

    install_cmd = 'conda'
    requirements = ['scikit-learn',
                    'pip::git+https://github.com/lorentzenchr/'
                    'scikit-learn@glm_newton_lsmr_only'
                    ]

    parameters = {
        'solver': [
            'newton-cg',
            'lbfgs',
            'sgd',
            'newton-lsmr',
            'newton-cholesky',
        ],
    }
    parameter_template = "{solver}"

    def set_objective(self, X, y, model, dataset_model):
        self.X, self.y, self.model, self.dataset_model = X, y, model, dataset_model

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=LineSearchWarning)
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='Line Search failed')

        if self.solver == 'sgd':
            self.clf = SGDClassifier(
                loss="log_loss", penalty=None, fit_intercept=False, tol=1e-10,
                random_state=42, learning_rate="optimal"
            )
        elif model == 'logreg':
            self.clf = LogisticRegression(
                solver=self.solver,
                penalty=None, fit_intercept=False, tol=1e-10
            )
        elif model == 'poisson':
            self.clf = PoissonRegressor(
                solver=self.solver, tol=1e-10, fit_intercept=False, max_iter=100,
                alpha=1e-4
            )

    def skip(self, X, y, model, dataset_model):
        if (model not in dataset_model or
            model not in ['logreg', 'poisson'] or
                (model == 'poisson' and self.solver in ['newton-cg', 'sgd'])):
            return True, f"{model} not suitable for this dataset"

        return False, None

    def run(self, n_iter):
        if self.solver == "sgd":
            n_iter += 1
        self.clf.max_iter = n_iter + 1
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return dict(beta=self.clf.coef_.flatten())
