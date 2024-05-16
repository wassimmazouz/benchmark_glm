from benchopt import BaseDataset

from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import LabelBinarizer


class Dataset(BaseDataset):

    name = "diabetes"

    install_cmd = 'conda'
    requirements = ['scikit-learn']

    def get_data(self):
        # Unlike libsvm[leukemia], this dataset corresponds to the whole
        # leukemia  data with train + test data (72 samples) and not just
        # the training set.
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        return dict(X=X, y=y)
