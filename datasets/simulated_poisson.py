from benchopt import BaseDataset
from benchopt import safe_import_context
from benchopt.datasets import make_correlated_data

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "simulatedpoisson"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'n_samples, n_features': [
            (1000, 500),
            (5000, 200),
            (200, 500)
        ],
        'rho': [0, 0.6],
        'random_state': [27],
        'sparsity': [True, False]
    }

    def get_data(self):

        rng = np.random.RandomState(self.random_state)
        if self.sparsity:
            X_density = 0.1
        else:
            X_density = 1.0

        X, y, _ = make_correlated_data(
            self.n_samples, self.n_features, rho=self.rho,
            X_density=X_density, random_state=rng
        )

        # Generate poisson variables
        mu = np.exp(y)
        y = rng.poisson(mu).astype(np.float64)

        return dict(X=X, y=y, dataset_model=['poisson'])
