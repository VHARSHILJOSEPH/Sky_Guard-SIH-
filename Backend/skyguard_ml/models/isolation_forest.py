from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def build_isolation_forest(
    n_estimators: int = 300,
    max_samples: str | int = "auto",
    contamination: float = "auto",
    max_features: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """Build an Isolation Forest pipeline.

    Includes RobustScaler for consistency with the imputer output,
    though tree-based models are largely scale-invariant.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(25, 75))),
            (
                "model",
                IsolationForest(
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    contamination=contamination,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
