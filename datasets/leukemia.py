from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelBinarizer


class Dataset(BaseDataset):

    name = "leukemia"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        X, y = fetch_openml("leukemia", return_X_y=True)
        X = X.to_numpy()
        binarizer = LabelBinarizer(neg_label=-1, pos_label=1)
        y = binarizer.fit_transform(y)[:, 0].astype(X.dtype)
        data = dict(X=X, y=y)

        return data
