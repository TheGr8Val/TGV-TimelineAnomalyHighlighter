"""
Per-event explainability notes for flagged anomalies.

Rather than SHAP (heavyweight dependency), we rank each event's features
by z-score vs. the population and format the top deviations into plain-
English analyst notes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Features to skip in explanations (redundant or low-signal on their own)
_SKIP_FEATURES = {"inter_event_delta_log"}

_DAYS = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]


def generate_notes(
    features: pd.DataFrame,
    top_n: int = 3,
    min_zscore: float = 1.5,
) -> pd.Series:
    """
    For every row in `features`, produce a human-readable string explaining
    which features deviated most from the population.

    Parameters
    ----------
    features : DataFrame produced by features.extract_features()
    top_n    : max number of contributing factors per note
    min_zscore : features below this z-score are omitted from the note

    Returns
    -------
    pd.Series of str, same index as `features`
    """
    pop_stats = _population_stats(features)
    notes: List[str] = []

    for idx in features.index:
        row = features.loc[idx]
        ranked = _rank_deviations(row, pop_stats)
        note = _format_note(row, ranked[:top_n], pop_stats, min_zscore)
        notes.append(note)

    return pd.Series(notes, index=features.index, name="anomaly_notes")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

PopStats = Dict[str, Dict[str, float]]


def _population_stats(features: pd.DataFrame) -> PopStats:
    stats: PopStats = {}
    for col in features.columns:
        if col in _SKIP_FEATURES:
            continue
        vals = features[col].dropna()
        std = float(vals.std()) if vals.std() > 0 else 1.0
        stats[col] = {
            "mean": float(vals.mean()),
            "std": std,
            "p10": float(vals.quantile(0.10)),
            "p50": float(vals.quantile(0.50)),
            "p90": float(vals.quantile(0.90)),
        }
    return stats


def _rank_deviations(
    row: pd.Series, pop_stats: PopStats
) -> List[Tuple[str, float, float]]:
    """Return [(feature, abs_zscore, raw_value), ...] sorted by |z| desc."""
    devs: List[Tuple[str, float, float]] = []
    for feat, s in pop_stats.items():
        val = float(row.get(feat, 0.0))
        z = abs((val - s["mean"]) / s["std"])
        devs.append((feat, z, val))
    devs.sort(key=lambda x: -x[1])
    return devs


def _format_note(
    row: pd.Series,
    top_devs: List[Tuple[str, float, float]],
    pop_stats: PopStats,
    min_zscore: float,
) -> str:
    parts: List[str] = []

    for feat, z, val in top_devs:
        if z < min_zscore:
            continue
        s = pop_stats.get(feat, {})

        if feat == "hour":
            parts.append(
                f"Unusual hour ({val:.0f}:00; typical {s['p10']:.0f}–{s['p90']:.0f}h)"
            )
        elif feat == "day_of_week":
            day_name = _DAYS[int(val)] if 0 <= int(val) <= 6 else str(int(val))
            parts.append(f"Low-frequency day ({day_name})")
        elif feat == "is_weekend" and val == 1.0:
            parts.append("Weekend activity")
        elif feat == "inter_event_delta_s":
            parts.append(
                f"Gap of {val:.0f}s before event (median {s['p50']:.0f}s)"
            )
        elif feat == "burst_rate_60s":
            parts.append(
                f"Burst: {val:.0f} events in 60s window (90th-pct: {s['p90']:.0f})"
            )
        elif feat == "event_type_rarity":
            pct = (1.0 - val) * 100
            parts.append(f"Rare event type (seen in {pct:.1f}% of timeline)")
        elif feat == "user_rarity":
            pct = (1.0 - val) * 100
            parts.append(f"Rare user account ({pct:.1f}% of events)")
        elif feat == "host_rarity":
            pct = (1.0 - val) * 100
            parts.append(f"Rare host ({pct:.1f}% of events)")
        elif feat == "hour_event_type_zscore":
            sign = "+" if val >= 0 else ""
            parts.append(
                f"Atypical hour for this event type ({sign}{val:.1f}σ)"
            )

    return "; ".join(parts) if parts else "Statistical outlier (composite score)"
