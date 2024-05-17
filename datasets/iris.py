# This dataset is intended for the multilogreg model

from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "iris"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        iris = load_iris()
        X1, y1 = iris.data, iris.target

        # Convert labels to one-hot encoding
        encoder = OneHotEncoder()
        y2 = encoder.fit_transform(y1.reshape(-1, 1)).toarray()

        X, X_test, y, y_test = train_test_split(X1, y2, test_size=0.2, random_state=42)

        data = dict(X=X, y=y)

        return data
