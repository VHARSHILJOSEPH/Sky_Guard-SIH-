from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def build_robust_covariance(
    contamination: float = 0.05,
    support_fraction: float | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build a Robust Covariance (Elliptic Envelope) pipeline.

    Uses RobustScaler for consistency. Note: EllipticEnvelope requires
    n_samples > n_features; the training code must ensure this constraint.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(25, 75))),
            (
                "model",
                EllipticEnvelope(
                    contamination=contamination,
                    support_fraction=support_fraction,
                    random_state=random_state,
                ),
            ),
        ]
    )
