"""
Ingest DFIR timeline CSVs into a normalized DataFrame.
Supports generic CSV (user-mapped columns) and Plaso l2t_csv format.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Canonical Plaso l2t_csv column order
_PLASO_COLS = [
    "date", "time", "timezone", "MACB", "source", "sourcetype",
    "type", "user", "host", "short", "desc", "version",
    "filename", "inode", "notes", "format", "extra",
]

# Subset that strongly indicates a Plaso header
_PLASO_MARKERS = {"MACB", "sourcetype", "inode"}

# Output schema shared by all loaders
NORMALIZED_COLS = ["timestamp", "event_type", "description", "user", "host", "filename"]


@dataclass
class TimelineConfig:
    """Column mapping + parsing options for generic CSV input."""

    timestamp_col: str = "timestamp"
    event_type_col: Optional[str] = "event_type"
    description_col: Optional[str] = "description"
    user_col: Optional[str] = None
    host_col: Optional[str] = None
    filename_col: Optional[str] = None
    timestamp_format: Optional[str] = None  # None → pandas auto-infer


def autodetect_and_load(
    path: str | Path,
    config: Optional[TimelineConfig] = None,
) -> pd.DataFrame:
    """
    Detect CSV format (Plaso or generic) and load into normalized form.
    Falls back to generic if the header doesn't match Plaso markers.
    """
    path = Path(path)
    header = _read_first_line(path)
    if _is_plaso_header(header):
        return load_plaso_csv(path)
    return load_generic_csv(path, config)


def load_generic_csv(
    path: str | Path,
    config: Optional[TimelineConfig] = None,
) -> pd.DataFrame:
    """Load a generic CSV timeline using the supplied column mapping."""
    if config is None:
        config = TimelineConfig()

    df = pd.read_csv(path, on_bad_lines="skip", low_memory=False)
    return _normalize_generic(df, config)


def load_plaso_csv(path: str | Path) -> pd.DataFrame:
    """Load a Plaso l2t_csv timeline."""
    # Plaso files may or may not include the header row; we detect and skip
    raw_header = _read_first_line(Path(path))
    has_header = _is_plaso_header(raw_header)

    df = pd.read_csv(
        path,
        names=_PLASO_COLS,
        skiprows=1 if has_header else 0,
        on_bad_lines="skip",
        low_memory=False,
    )
    return _normalize_plaso(df)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return fh.readline().strip()


def _is_plaso_header(header: str) -> bool:
    cols = {c.strip().strip('"') for c in header.split(",")}
    return len(_PLASO_MARKERS & cols) >= 2


def _normalize_generic(df: pd.DataFrame, cfg: TimelineConfig) -> pd.DataFrame:
    out = pd.DataFrame()

    out["timestamp"] = pd.to_datetime(
        df[cfg.timestamp_col],
        format=cfg.timestamp_format,
        utc=True,
        errors="coerce",
    )

    def _col(name: Optional[str], default: str = "") -> pd.Series:
        if name and name in df.columns:
            return df[name].fillna(default).astype(str)
        return pd.Series(default, index=df.index)

    out["event_type"] = _col(cfg.event_type_col, "UNKNOWN")
    out["description"] = _col(cfg.description_col)
    out["user"] = _col(cfg.user_col)
    out["host"] = _col(cfg.host_col)
    out["filename"] = _col(cfg.filename_col)

    return _finalize(out)


def _normalize_plaso(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()

    date_time_str = (
        df["date"].astype(str).str.strip()
        + " "
        + df["time"].astype(str).str.strip()
    )
    out["timestamp"] = pd.to_datetime(
        date_time_str,
        format="%m/%d/%Y %H:%M:%S",
        utc=True,
        errors="coerce",
    )

    out["event_type"] = (
        df["type"].fillna(df["sourcetype"]).fillna("UNKNOWN").astype(str)
    )
    out["description"] = (
        df["desc"].fillna(df["short"]).fillna("").astype(str)
    )
    out["user"] = df["user"].fillna("").astype(str)
    out["host"] = df["host"].fillna("").astype(str)
    out["filename"] = df["filename"].fillna("").astype(str)

    return _finalize(out)


def _finalize(out: pd.DataFrame) -> pd.DataFrame:
    """Drop bad timestamps, sort chronologically, reset index."""
    return (
        out.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
