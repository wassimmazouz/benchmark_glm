from benchopt import BaseSolver, safe_import_context

# Protect the import with `safe_import_context()`.
with safe_import_context() as import_ctx:
    import torch
    from benchmark_utils.obj_helper import objective_function_linreg, objective_function_logreg


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'Adam'

    parameters = {
        'lr': [0.001],
        'weight_decay': [0.5],
        'amsgrad': [False],
    }

    requirements = ['torch']

    def set_objective(self, X, y, model, dataset_model):
        self.X, self.y, self.model = X, y, model
        self.dataset_model = dataset_model

    def skip(self, X, y, model, dataset_model):
        if model not in dataset_model or model not in ['linreg', 'logreg']:
            return True, f"{model} not suitable for this dataset"
        return False, None

    def run(self, n_iter):
        X_torch = torch.tensor(self.X, dtype=torch.float32)
        y_torch = torch.tensor(self.y, dtype=torch.float32)
        beta_torch = torch.zeros(X_torch.shape[1], requires_grad=True)

        if self.model == 'logreg':
            def objective(beta):
                return objective_function_logreg(X_torch, y_torch, beta).sum()

        if self.model == 'linreg':
            def objective(beta):
                return objective_function_linreg(X_torch, y_torch, beta).sum()

        optimizer = torch.optim.Adam([beta_torch], lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     amsgrad=self.amsgrad)

        for _ in range(n_iter):
            optimizer.zero_grad()
            loss = objective(beta_torch)
            loss.backward()
            optimizer.step()

        self.beta = beta_torch.detach().numpy()

    def get_result(self):
        return dict(beta=self.beta)
