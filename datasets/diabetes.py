from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import load_diabetes
    import numpy as np


class Dataset(BaseDataset):

    name = "diabetes"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        X, y = load_diabetes(return_X_y=True)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        return dict(X=X, y=y, dataset_model=['linreg'])
