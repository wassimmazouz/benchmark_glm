
from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.datasets import fetch_openml
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
    from sklearn.preprocessing import StandardScaler, KBinsDiscretizer


class Dataset(BaseDataset):

    name = "insurance"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
    }

    def get_data(self):
        df = fetch_openml(data_id=41214, as_frame=True).frame
        df["Frequency"] = df["ClaimNb"] / df["Exposure"]
        log_scale_transformer = make_pipeline(
            FunctionTransformer(np.log, validate=False), StandardScaler()
        )
        linear_model_preprocessor = ColumnTransformer(
            [
                ("passthrough_numeric", "passthrough", ["BonusMalus"]),
                (
                    "binned_numeric",
                    KBinsDiscretizer(n_bins=10, subsample=None),
                    ["VehAge", "DrivAge"],
                ),
                ("log_scaled_numeric", log_scale_transformer, ["Density"]),
                (
                    "onehot_categorical",
                    OneHotEncoder(),
                    ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
                ),
            ],
            remainder="drop",
        )

        X = linear_model_preprocessor.fit_transform(df)
        y = df["Frequency"].values

        return dict(X=X, y=y, dataset_model=['poisson'])
