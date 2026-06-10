"""
Ensemble anomaly detector: IsolationForest + LocalOutlierFactor.

Design note: LOF with novelty=False only scores training data via
negative_outlier_factor_, so fit_score() is the primary entry point —
it fits and scores in one pass over the same feature matrix, which is
the correct usage pattern for single-timeline DFIR analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


@dataclass
class DetectorConfig:
    contamination: float = 0.05      # Expected fraction of anomalous events
    lof_n_neighbors: int = 20        # k for LOF; auto-clipped to n_samples - 1
    if_weight: float = 0.5           # IsolationForest weight in ensemble
    lof_weight: float = 0.5          # LOF weight in ensemble
    flag_threshold: float = 0.65     # Score ≥ threshold → flagged anomaly
    random_state: int = 42


class AnomalyDetector:
    """
    Fit on a feature matrix and return anomaly scores in [0, 1].
    Higher score = more anomalous.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._scaler = StandardScaler()
        self._scores: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def fit_score(self, features: pd.DataFrame) -> pd.Series:
        """
        Fit IsolationForest + LOF on `features` and return ensemble scores.
        Scores are in [0, 1]; call flag() afterwards to get the boolean mask.
        """
        cfg = self.config
        X = self._scaler.fit_transform(features.values)

        # IsolationForest: decision_function returns values where more negative
        # = more anomalous; negate so higher = more anomalous.
        iso = IsolationForest(
            contamination=cfg.contamination,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        if_raw = -iso.fit(X).decision_function(X)

        # LOF with novelty=False: negative_outlier_factor_ is set after fit.
        # More negative = more anomalous; negate for consistent direction.
        k = min(cfg.lof_n_neighbors, len(X) - 1)
        lof = LocalOutlierFactor(
            n_neighbors=k,
            contamination=cfg.contamination,
            novelty=False,
            n_jobs=-1,
        )
        lof.fit(X)
        lof_raw = -lof.negative_outlier_factor_

        # Normalize each scorer to [0, 1] then weight-average
        if_norm = _minmax(if_raw)
        lof_norm = _minmax(lof_raw)
        total_w = cfg.if_weight + cfg.lof_weight
        ensemble = (cfg.if_weight * if_norm + cfg.lof_weight * lof_norm) / total_w

        self._scores = pd.Series(ensemble, index=features.index, name="anomaly_score")
        return self._scores

    def flag(self, threshold: Optional[float] = None) -> pd.Series:
        """
        Return a boolean Series: True where anomaly_score >= threshold.
        Must call fit_score() first.
        """
        if self._scores is None:
            raise RuntimeError("Call fit_score() before flag()")
        t = threshold if threshold is not None else self.config.flag_threshold
        return (self._scores >= t).rename("is_anomaly")

    @property
    def scores(self) -> Optional[pd.Series]:
        return self._scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)
