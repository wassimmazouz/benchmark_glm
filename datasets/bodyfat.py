from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    import numpy as np


class Dataset(BaseDataset):

    name = "bodyfat"

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    def get_data(self):
        X, y = fetch_libsvm("bodyfat")
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

        return dict(X=X, y=y)
