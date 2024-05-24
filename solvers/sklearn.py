import warnings


from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from scipy.optimize.linesearch import LineSearchWarning


class Solver(BaseSolver):
    name = 'sklearn'

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    parameters = {
        'solver': [
            'newton-cg',
            'lbfgs',
            'sgd',
        ],
    }
    parameter_template = "{solver}"

    def set_objective(self, X, y, model):
        if model != 'multilogreg' and model != 'logreg':
            print("Please choose a logistic model to use this solver")
            # Stop execution by raising an exception
            raise ValueError("Wrong model for solver")

        self.X, self.y = X, y

        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        warnings.filterwarnings('ignore', category=LineSearchWarning)
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='Line Search failed')

        if self.solver == 'sgd':
            self.clf = SGDClassifier(
                loss="log_loss", penalty=None, fit_intercept=False, tol=1e-10,
                random_state=42, learning_rate="optimal"
            )
        else:
            self.clf = LogisticRegression(
                solver=self.solver,
                penalty=None, fit_intercept=False, tol=1e-10
            )

    def run(self, n_iter):
        if self.solver == "sgd":
            n_iter += 1
        self.clf.max_iter = n_iter
        self.clf.fit(self.X, self.y)

    def get_result(self):
        return dict(beta=self.clf.coef_.flatten())
