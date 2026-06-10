# TGV Timeline Anomaly Highlighter

**ML-powered anomaly detection for DFIR timelines.**

Ingest a DFIR timeline CSV → get back every event scored for statistical anomaly, with plain-English explainability notes for each flagged event.

Built with `scikit-learn` (IsolationForest + LocalOutlierFactor ensemble). No LLM / API key required.

---

## Quick start

```bash
pip install -e ".[dev]"
```

```python
import timeline_anomaly as ta

df = ta.analyze("examples/sample_generic.csv")
flagged = df[df["is_anomaly"]]
print(flagged[["timestamp", "event_type", "anomaly_score", "anomaly_notes"]])
```

---

## Supported input formats

| Format | How to use |
|--------|-----------|
| **Generic CSV** | Any CSV with a timestamp column + optional event_type, description, user, host, filename columns. Column names are configurable via `TimelineConfig`. |
| **Plaso l2t_csv** | Autodetected from the `MACB`, `sourcetype`, `inode` header markers. Produced by `psort.py -o l2tcsv`. |

Both formats are autodetected by default:

```python
df = ta.autodetect_and_load("my_timeline.csv")
```

For a generic CSV with non-default column names:

```python
from timeline_anomaly import TimelineConfig, autodetect_and_load

cfg = TimelineConfig(
    timestamp_col="datetime",
    event_type_col="activity",
    user_col="account",
    host_col="machine",
)
df = autodetect_and_load("custom.csv", config=cfg)
```

---

## How it works

### Feature engineering (`features.py`)

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `day_of_week` | 0=Monday … 6=Sunday |
| `is_weekend` | 1 if Saturday/Sunday |
| `inter_event_delta_s` | Seconds since previous event |
| `inter_event_delta_log` | log1p of the above (handles extreme outliers) |
| `burst_rate_60s` | Events in the 60-second window ending at this event |
| `event_type_rarity` | 1 − frequency fraction of this event type |
| `user_rarity` | 1 − frequency fraction of this user |
| `host_rarity` | 1 − frequency fraction of this host |
| `hour_event_type_zscore` | Z-score of hour vs. this event type's hour distribution |

### Detection (`detector.py`)

Ensemble of two unsupervised scorers, each normalized to [0, 1]:

- **IsolationForest** — isolation-based outlier scoring; robust to high-dimensional data
- **LocalOutlierFactor** — density-based; catches local clusters that appear as outliers in context

Scores are blended 50/50 by default. Events with ensemble score ≥ 0.65 are flagged.

### Explainability (`explain.py`)

For each event, the top-N features with the highest z-score relative to the population are rendered as plain-English notes, e.g.:

> Unusual hour (3:00; typical 9–17h); Burst: 6 events in 60s window (90th-pct: 1); Rare event type (seen in 1.5% of timeline)

---

## Configuration

```python
from timeline_anomaly import DetectorConfig, AnomalyDetector

cfg = DetectorConfig(
    contamination=0.05,      # Expected fraction of anomalies (default 0.05)
    lof_n_neighbors=20,      # k for LOF
    if_weight=0.5,           # IsolationForest ensemble weight
    lof_weight=0.5,          # LOF ensemble weight
    flag_threshold=0.65,     # Score ≥ this → is_anomaly = True
)
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Project structure

```
timeline_anomaly/
├── ingest.py    # CSV parsers (generic + Plaso l2t_csv)
├── features.py  # Feature engineering
├── detector.py  # IsolationForest + LOF ensemble
├── explain.py   # Per-event explainability notes
└── __init__.py  # Public API + analyze() convenience function

examples/
├── sample_generic.csv   # Realistic DFIR scenario (generic format)
└── sample_plaso.csv     # Same scenario in Plaso l2t_csv format

tests/
├── conftest.py          # Shared fixtures
├── test_ingest.py
├── test_features.py
└── test_detector.py
```

---

## Embedded anomalies in sample data

The sample timelines include a deliberate insider-threat scenario for demonstration:

| Timestamp | What | Why it's anomalous |
|-----------|------|--------------------|
| 2024-03-14 03:11–03:13 | Mass Finance share access + USB copy | 3 AM activity, burst of 7 events in 2 minutes, rare event types |
| 2024-03-14 22:47–22:48 | Registry run key + suspicious process + external connection | Late-night, rare event types (REGISTRY_WRITE, NETWORK_CONN), unusual host activity |

---

## Roadmap

- [ ] CLI (`timeline-anomaly score timeline.csv`)
- [ ] Self-contained HTML report with annotated interactive timeline
- [ ] Medium article walkthrough
