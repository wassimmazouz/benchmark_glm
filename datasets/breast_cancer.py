from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import LabelBinarizer
    import numpy as np


class Dataset(BaseDataset):

    name = "bcancer"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        X, y = load_breast_cancer(return_X_y=True)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        y = binarizer.fit_transform(y)[:, 0].astype(X.dtype)

        return dict(X=X, y=y, dataset_model=['logreg'])
