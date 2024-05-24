from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    parameters = {
        'n_samples, n_features': [
            (1000, 500),
            (5000, 200),
            (200, 500),
        ],
        'random_state': [27],
        'binary': [True, False]
    }

    def __init__(self, n_samples=10, n_features=50, random_state=42, binary=False):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.binary = binary

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        beta = rng.randn(self.n_features)

        X = rng.randn(self.n_samples, self.n_features)
        if self.binary:
            y = np.sign(X @ beta)
        else:
            y = X @ beta

        return dict(X=X, y=y)
