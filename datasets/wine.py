# This dataset is intended for the multilogreg model

from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import load_wine
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split


class Dataset(BaseDataset):

    name = "wine"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        X, y1 = load_wine(return_X_y=True)

        # Convert labels to one-hot encoding
        encoder = OneHotEncoder()
        y = encoder.fit_transform(y1.reshape(-1, 1)).toarray()

        return dict(X=X, y=y)
