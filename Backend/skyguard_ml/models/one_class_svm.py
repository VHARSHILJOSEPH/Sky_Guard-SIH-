from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import OneClassSVM


def build_one_class_svm(
    kernel: str = "rbf",
    nu: float = 0.05,
    gamma: str | float = "scale",
) -> Pipeline:
    """Build a One-Class SVM pipeline.

    Uses RobustScaler (median/IQR) instead of StandardScaler so that
    the scaler itself is not skewed by outliers in the training data.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(25, 75))),
            (
                "model",
                OneClassSVM(
                    kernel=kernel,
                    nu=nu,
                    gamma=gamma,
                ),
            ),
        ]
    )
