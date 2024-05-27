# This dataset is intended for the multilogreg model

from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import OneHotEncoder


class Dataset(BaseDataset):

    name = "iris"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        iris = load_iris()
        X, y = iris.data, iris.target

        # Convert labels to one-hot encoding
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

        return dict(X=X, y=y)
