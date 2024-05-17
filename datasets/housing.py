from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_california_housing
    import numpy as np


class Dataset(BaseDataset):

    name = "housing"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        X1, y1 = fetch_california_housing(return_X_y=True)

        mean = np.mean(X1, axis=0)
        std = np.std(X1, axis=0)

        X = (X1 - mean) / std
        y = y1

        return dict(X=X, y=y)
