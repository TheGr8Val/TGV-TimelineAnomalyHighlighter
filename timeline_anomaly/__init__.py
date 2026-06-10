"""
timeline_anomaly — ML-powered anomaly highlighter for DFIR timelines.

Quick start
-----------
>>> import timeline_anomaly as ta
>>> df = ta.analyze("my_timeline.csv")
>>> print(df[df["is_anomaly"]][["timestamp", "event_type", "anomaly_score", "anomaly_notes"]])
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .ingest import (
    TimelineConfig,
    autodetect_and_load,
    load_generic_csv,
    load_plaso_csv,
)
from .features import FEATURE_COLS, extract_features
from .detector import AnomalyDetector, DetectorConfig
from .explain import generate_notes

__version__ = "0.1.0"
__all__ = [
    "analyze",
    "TimelineConfig",
    "DetectorConfig",
    "AnomalyDetector",
    "autodetect_and_load",
    "load_generic_csv",
    "load_plaso_csv",
    "extract_features",
    "generate_notes",
    "FEATURE_COLS",
]


def analyze(
    path: str | Path,
    config: Optional[TimelineConfig] = None,
    detector_config: Optional[DetectorConfig] = None,
    flag_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    One-shot pipeline: load → extract features → detect → explain.

    Parameters
    ----------
    path             : Path to a timeline CSV (generic or Plaso l2t_csv).
    config           : Column-mapping config for generic CSVs.
    detector_config  : Override IsolationForest/LOF hyperparameters.
    flag_threshold   : Score cutoff for is_anomaly (default 0.65).

    Returns
    -------
    The normalized timeline DataFrame enriched with three new columns:
      - anomaly_score  float [0, 1], higher = more suspicious
      - is_anomaly     bool, True when score >= flag_threshold
      - anomaly_notes  str, plain-English explanation of top drivers
    """
    df = autodetect_and_load(path, config)
    features = extract_features(df)

    detector = AnomalyDetector(detector_config)
    scores = detector.fit_score(features)
    flags = detector.flag(flag_threshold)
    notes = generate_notes(features)

    df["anomaly_score"] = scores.round(4)
    df["is_anomaly"] = flags
    df["anomaly_notes"] = notes

    return df
