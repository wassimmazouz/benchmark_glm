from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from libsvmdata import fetch_libsvm
    import numpy as np


class Dataset(BaseDataset):

    name = "libsvm"

    parameters = {
        "dataset": ["bodyfat"],
    }

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    def __init__(self, dataset="bodyfat"):
        self.dataset = dataset
        self.X, self.y = None, None

    def get_data(self):

        if self.X is None:
            X1, y1 = fetch_libsvm(self.dataset)
            mean = np.mean(X1, axis=0)
            std = np.std(X1, axis=0)

            self.X = (X1 - mean) / std
            self.y = y1

        if self.dataset == "SUSY":
            self.y = 2 * (self.y > 0) - 1

        data = dict(X=self.X, y=self.y)

        return data
