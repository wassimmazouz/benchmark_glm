from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import LabelBinarizer
    import pandas as pd


class Dataset(BaseDataset):

    name = "housing"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        X1, y1 = fetch_california_housing(return_X_y=True)

        
        X = X1[:]
        y = y1[:]
        return dict(X=X ,y=y)
