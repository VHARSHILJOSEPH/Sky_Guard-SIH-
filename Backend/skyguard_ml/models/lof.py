"""
LOF (Local Outlier Factor) anomaly detector.

Uses RobustScaler (median/IQR) instead of StandardScaler so the scaler
itself is not skewed by the very outliers we are trying to detect.
Parallel processing via n_jobs=-1.
"""

from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


def build_lof(
    n_neighbors: int = 20,
    contamination: float = 0.03,
    metric: str = "minkowski",
    **kwargs,
):
    """Build an LOF anomaly-detection pipeline.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbours used by LOF (default 20).
    contamination : float
        Expected proportion of outliers in the training data.
    metric : str
        Distance metric for the nearest-neighbour lookup.
    **kwargs
        Extra keyword arguments forwarded to ``LocalOutlierFactor``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Imputer → RobustScaler → LOF (novelty mode).
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(quantile_range=(25, 75)),
        LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric,
            novelty=True,
            n_jobs=-1,
            algorithm="auto",
            leaf_size=30,
            **kwargs,
        ),
    )


# ── Convenience aliases for model selection ─────────────────────────────────
build_lof_default = lambda: build_lof(n_neighbors=20, contamination=0.03)
build_lof_sensitive = lambda: build_lof(n_neighbors=30, contamination=0.05)
build_lof_strict = lambda: build_lof(n_neighbors=15, contamination=0.02)
