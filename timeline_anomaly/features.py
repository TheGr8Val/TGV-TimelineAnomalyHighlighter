"""
Feature engineering for DFIR timeline anomaly detection.
Transforms a normalized timeline DataFrame into a numeric feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Feature columns produced by extract_features(), in stable order
FEATURE_COLS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "inter_event_delta_s",
    "inter_event_delta_log",
    "burst_rate_60s",
    "event_type_rarity",
    "user_rarity",
    "host_rarity",
    "hour_event_type_zscore",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ML features from a normalized timeline DataFrame.

    Input columns used: timestamp, event_type, user, host.
    Returns a float64 DataFrame with the same index, no NaNs.
    """
    feats = pd.DataFrame(index=df.index)
    ts = df["timestamp"]

    # --- Temporal ---
    feats["hour"] = ts.dt.hour.astype(float)
    feats["day_of_week"] = ts.dt.dayofweek.astype(float)
    feats["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float)

    # --- Inter-event delta (time gap to previous event) ---
    delta_s = ts.diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
    feats["inter_event_delta_s"] = delta_s
    feats["inter_event_delta_log"] = np.log1p(delta_s)

    # --- Burst rate: number of events in the 60-second window ending at each event ---
    feats["burst_rate_60s"] = _rolling_event_count(ts, window_seconds=60).astype(float)

    # --- Rarity features (1 = completely unique, 0 = most common) ---
    feats["event_type_rarity"] = _rarity(df["event_type"])
    feats["user_rarity"] = _rarity(df["user"])
    feats["host_rarity"] = _rarity(df["host"])

    # --- Hour z-score within each event_type group ---
    feats["hour_event_type_zscore"] = _group_zscore(feats["hour"], df["event_type"])

    return feats[FEATURE_COLS].fillna(0.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_event_count(timestamps: pd.Series, window_seconds: int = 60) -> pd.Series:
    """
    Two-pointer O(n) sliding window: for each event counts how many events
    (including itself) fall within [t - window_seconds, t].
    """
    ts_sorted = timestamps.sort_values()
    arr = ts_sorted.astype(np.int64).values // 10 ** 9  # epoch seconds
    n = len(arr)
    counts = np.empty(n, dtype=np.int32)
    left = 0
    for right in range(n):
        while arr[right] - arr[left] > window_seconds:
            left += 1
        counts[right] = right - left + 1
    return pd.Series(counts, index=ts_sorted.index).reindex(timestamps.index).fillna(1)


def _rarity(series: pd.Series) -> pd.Series:
    """
    Rarity = 1 − frequency_fraction.
    Empty / unknown strings are treated as neutral (0.5) so they don't
    skew rarity when a column isn't populated.
    """
    freq = series.value_counts(normalize=True)
    rarity = 1.0 - series.map(freq)
    rarity[series.isin(["", "N/A", "nan", "None"])] = 0.5
    return rarity.fillna(0.5).astype(np.float64)


def _group_zscore(values: pd.Series, groups: pd.Series) -> pd.Series:
    """
    Z-score of each value relative to its group's distribution.
    Groups with fewer than 3 members get score 0 (not enough data to judge).
    """
    result = pd.Series(0.0, index=values.index)
    for group_val, idx in groups.groupby(groups).groups.items():
        grp = values[idx]
        if len(grp) < 3:
            continue
        std = grp.std()
        if std > 0:
            result[idx] = (grp - grp.mean()) / std
    return result.fillna(0.0)
