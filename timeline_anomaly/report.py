"""
Self-contained HTML report generator for analyzed DFIR timelines.
Single-file output — no CDN, no external dependencies.
"""

from __future__ import annotations

import html as _html
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# ─── CSS ─────────────────────────────────────────────────────────────────────

_CSS = """\
:root {
    --bg:     #0A0A12;
    --bg2:    #11111C;
    --bg3:    #1A1A28;
    --bg4:    #1E1E30;
    --purple: #A855F7;
    --purple-mid: #7C3AED;
    --pink:   #EC4899;
    --indigo: #818CF8;
    --text:   #F1F0FF;
    --muted:  #9B8FCE;
    --dim:    #5A5280;
    --border: rgba(168,85,247,0.22);
    --red:    #EF4444;
    --orange: #F97316;
    --amber:  #F59E0B;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 13px;
    line-height: 1.6;
    padding-bottom: 48px;
}
code, .mono {
    font-family: 'Cascadia Code', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}
.container { max-width: 1280px; margin: 0 auto; padding: 0 24px; }

/* ── Header ──────────────────────────────────────────────────────────── */
.page-header {
    background: linear-gradient(135deg, rgba(168,85,247,0.10) 0%, rgba(236,72,153,0.06) 100%);
    border-bottom: 1px solid var(--border);
    padding: 28px 32px 22px;
    margin-bottom: 4px;
}
.header-brand {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--purple);
    margin-bottom: 6px;
}
.header-title { font-size: 22px; font-weight: 800; color: var(--text); margin-bottom: 4px; }
.header-meta  { font-size: 12px; color: var(--muted); }

/* ── Section chrome ──────────────────────────────────────────────────── */
.section { margin-top: 28px; }
.section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.10em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}
.section-sub { font-size: 12px; color: var(--dim); margin-bottom: 8px; }

/* ── Summary cards ───────────────────────────────────────────────────── */
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
}
.stat-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
}
.stat-card.hi { border-color: rgba(236,72,153,0.35); }
.stat-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.stat-value { font-size: 28px; font-weight: 800; color: var(--text); line-height: 1; }
.stat-card.hi .stat-value { color: var(--pink); }
.stat-sub { font-size: 11px; color: var(--dim); margin-top: 3px; }
.stat-value.small { font-size: 14px; padding-top: 4px; }

/* ── Score strip ─────────────────────────────────────────────────────── */
.score-strip {
    display: flex;
    align-items: flex-end;
    height: 64px;
    gap: 1px;
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    overflow: hidden;
}
.strip-bar {
    flex: 1 1 0;
    min-width: 2px;
    border-radius: 2px 2px 0 0;
    opacity: 0.82;
    cursor: default;
    transition: opacity 0.12s;
}
.strip-bar:hover { opacity: 1; }

/* ── Top anomalies ───────────────────────────────────────────────────── */
.anomaly-card {
    background: var(--bg2);
    border: 1px solid rgba(239,68,68,0.18);
    border-left: 3px solid;
    border-radius: 8px;
    padding: 13px 16px;
    margin-bottom: 10px;
}
.anomaly-card.tier-critical { border-left-color: var(--red); }
.anomaly-card.tier-high     { border-left-color: var(--orange); }
.anomaly-card.tier-flagged  { border-left-color: var(--amber); }
.anomaly-header { display: flex; align-items: center; gap: 12px; margin-bottom: 5px; flex-wrap: wrap; }
.score-badge {
    font-size: 12px; font-weight: 800;
    font-family: 'Cascadia Code', monospace;
    padding: 2px 8px; border-radius: 4px;
    background: rgba(239,68,68,0.14); color: var(--red);
    flex-shrink: 0;
}
.score-badge.tier-high    { background: rgba(249,115,22,0.14); color: var(--orange); }
.score-badge.tier-flagged { background: rgba(245,158,11,0.14); color: var(--amber); }
.anomaly-ts    { font-size: 12px; color: var(--dim); font-family: monospace; }
.anomaly-etype { font-weight: 700; color: var(--text); font-size: 13px; }
.anomaly-notes { font-size: 12px; color: var(--muted); margin-top: 4px; line-height: 1.5; }
.anomaly-desc  { font-size: 12px; color: var(--dim); margin-top: 2px; }

/* ── Controls ────────────────────────────────────────────────────────── */
.controls { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
.btn {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    cursor: pointer;
    font-size: 12px; font-weight: 600;
    padding: 6px 14px;
    transition: border-color 0.14s, background 0.14s;
}
.btn:hover  { background: var(--bg4); border-color: var(--purple); }
.btn.active { background: rgba(168,85,247,0.14); border-color: var(--purple); color: var(--purple); }
.result-count { font-size: 12px; color: var(--dim); margin-left: auto; }

/* ── Timeline table ──────────────────────────────────────────────────── */
.timeline-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; }
table { border-collapse: collapse; width: 100%; min-width: 640px; }
thead th {
    background: var(--bg3);
    color: var(--muted);
    font-size: 11px; font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    padding: 10px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}
tbody tr { border-bottom: 1px solid rgba(168,85,247,0.07); }
tbody tr:hover td { background: rgba(168,85,247,0.04) !important; }
tbody td { padding: 9px 14px; vertical-align: middle; font-size: 12px; }
.row-normal   td:first-child { border-left: 2px solid transparent; }
.row-flagged  td:first-child { border-left: 2px solid var(--amber); }
.row-high     td:first-child { border-left: 2px solid var(--orange); }
.row-critical td:first-child { border-left: 2px solid var(--red); }
.row-flagged  { background: rgba(245,158,11,0.03); }
.row-high     { background: rgba(249,115,22,0.05); }
.row-critical { background: rgba(239,68,68,0.07); }
.ts-cell { color: var(--dim); font-family: monospace; white-space: nowrap; }
.etype-badge {
    display: inline-block;
    background: rgba(168,85,247,0.10);
    border: 1px solid rgba(168,85,247,0.18);
    border-radius: 4px;
    color: var(--indigo);
    font-size: 11px; font-weight: 600;
    padding: 1px 7px;
    white-space: nowrap;
}
.score-cell { min-width: 120px; white-space: nowrap; }
.score-bar-wrap { display: flex; align-items: center; gap: 8px; }
.score-track {
    flex: 1; height: 5px; background: var(--bg3);
    border-radius: 3px; overflow: hidden; min-width: 56px;
}
.score-fill  { height: 100%; border-radius: 3px; }
.score-num   { font-family: monospace; font-size: 12px; font-weight: 700; width: 40px; text-align: right; flex-shrink: 0; }
.notes-cell  { color: var(--muted); font-size: 11px; max-width: 340px; line-height: 1.5; }
.notes-empty { color: var(--dim); font-size: 11px; }
.desc-cell   { color: var(--muted); max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ── Footer ──────────────────────────────────────────────────────────── */
.page-footer {
    margin-top: 40px;
    padding: 18px 32px;
    border-top: 1px solid var(--border);
    display: flex; align-items: center;
    justify-content: space-between;
    flex-wrap: wrap; gap: 8px;
}
.footer-brand {
    font-size: 12px; font-weight: 800;
    background: linear-gradient(90deg, var(--purple), var(--pink));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.footer-meta { font-size: 11px; color: var(--dim); }

@media print {
    body { background: #fff; color: #000; }
    .btn, .controls { display: none !important; }
    .page-header { background: #fff; border-bottom: 1px solid #ccc; }
}
"""

# ─── JS ──────────────────────────────────────────────────────────────────────

_JS = """\
(function () {
  var flaggedOnly = false;
  var rows = [];
  function init() {
    rows = Array.from(document.querySelectorAll('tr[data-flagged]'));
    updateCount();
  }
  window.toggleFilter = function () {
    flaggedOnly = !flaggedOnly;
    var btn = document.getElementById('filter-btn');
    if (flaggedOnly) {
      btn.textContent = 'Show All Events';
      btn.classList.add('active');
      rows.forEach(function (r) {
        r.style.display = r.dataset.flagged === 'true' ? '' : 'none';
      });
    } else {
      btn.textContent = 'Flagged Only';
      btn.classList.remove('active');
      rows.forEach(function (r) { r.style.display = ''; });
    }
    updateCount();
  };
  function updateCount() {
    var visible = flaggedOnly
      ? rows.filter(function (r) { return r.dataset.flagged === 'true'; }).length
      : rows.length;
    var el = document.getElementById('row-count');
    if (el) el.textContent = 'Showing ' + visible + ' of ' + rows.length + ' events';
  }
  document.addEventListener('DOMContentLoaded', init);
})();
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _e(s: object) -> str:
    return _html.escape("" if s is None else str(s))


def _clean(s: object) -> str:
    v = str(s) if s is not None else ""
    return "" if v in ("nan", "None", "NaN") else v


def _score_color(score: float) -> str:
    if score >= 0.80:
        return "#EF4444"
    if score >= 0.65:
        return "#F97316"
    if score >= 0.50:
        return "#F59E0B"
    return "#7C3AED"


def _tier(score: float, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "normal"
    if score >= 0.85:
        return "critical"
    if score >= 0.75:
        return "high"
    return "flagged"


def _fmt_ts(ts) -> str:
    if pd.isna(ts):
        return ""
    try:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)[:19]


def _has_content(series: pd.Series) -> bool:
    return (
        series.astype(str).str.strip()
        .replace({"nan": "", "None": "", "NaN": ""})
        .ne("")
        .any()
    )


# ─── Section builders ─────────────────────────────────────────────────────────

def _header(title: str, generated_at: str) -> str:
    return (
        '<header class="page-header">\n'
        '  <div class="container">\n'
        f'    <div class="header-brand">thegr8val · DFIR ML Tool</div>\n'
        f'    <h1 class="header-title">{_e(title)}</h1>\n'
        f'    <p class="header-meta">Generated {_e(generated_at)}'
        " · Timeline Anomaly Highlighter v0.1.0</p>\n"
        "  </div>\n"
        "</header>"
    )


def _summary_cards(
    total: int, flagged: int, flag_rate: float, max_score: float, time_range: str
) -> str:
    return f"""\
<div class="section">
  <div class="summary-grid">
    <div class="stat-card">
      <div class="stat-label">Total Events</div>
      <div class="stat-value">{total:,}</div>
    </div>
    <div class="stat-card hi">
      <div class="stat-label">Flagged</div>
      <div class="stat-value">{flagged:,}</div>
      <div class="stat-sub">{flag_rate:.1f}% of timeline</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Max Score</div>
      <div class="stat-value">{max_score:.3f}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Time Range</div>
      <div class="stat-value small mono">{_e(time_range)}</div>
    </div>
  </div>
</div>"""


def _score_strip(df: pd.DataFrame) -> str:
    max_h = 44
    bars = []
    for _, row in df.iterrows():
        score = float(row["anomaly_score"])
        h = max(2, int(score * max_h))
        color = _score_color(score)
        ts_str = _fmt_ts(row["timestamp"])
        et = _e(_clean(row.get("event_type", "")))
        bars.append(
            f'<div class="strip-bar" style="height:{h}px;background:{color}" '
            f'title="{ts_str} | {et} | score: {score:.3f}"></div>'
        )
    return (
        '<div class="section">\n'
        '  <div class="section-title">Score Distribution</div>\n'
        '  <div class="section-sub">Each bar = one event · Height = anomaly score · Hover for details</div>\n'
        f'  <div class="score-strip">{"".join(bars)}</div>\n'
        "</div>"
    )


def _top_anomalies(df: pd.DataFrame, n: int = 5) -> str:
    top = df[df["is_anomaly"]].nlargest(n, "anomaly_score")
    if top.empty:
        return ""

    cards = []
    for _, row in top.iterrows():
        score = float(row["anomaly_score"])
        t = _tier(score, True)
        badge_cls = f"tier-{t}"
        ts = _fmt_ts(row["timestamp"])
        et = _e(_clean(row.get("event_type", "")))
        desc = _e(_clean(row.get("description", "")))
        notes = _e(_clean(row.get("anomaly_notes", "")))

        desc_html = f'  <div class="anomaly-desc">{desc}</div>\n' if desc else ""
        notes_html = f'  <div class="anomaly-notes">{notes}</div>\n' if notes else ""

        cards.append(
            f'<div class="anomaly-card tier-{t}">\n'
            f'  <div class="anomaly-header">\n'
            f'    <span class="score-badge {badge_cls}">{score:.3f}</span>\n'
            f'    <span class="anomaly-ts">{ts}</span>\n'
            f'    <span class="anomaly-etype">{et}</span>\n'
            f'  </div>\n'
            f"{desc_html}"
            f"{notes_html}"
            f"</div>"
        )

    return (
        '<div class="section">\n'
        '  <div class="section-title">Top Flagged Events</div>\n'
        + "\n".join(cards)
        + "\n</div>"
    )


def _table_section(df: pd.DataFrame) -> str:
    has_desc = _has_content(df.get("description", pd.Series(dtype=str)))
    has_user = _has_content(df.get("user", pd.Series(dtype=str)))
    has_host = _has_content(df.get("host", pd.Series(dtype=str)))

    extra_ths = ""
    if has_desc:
        extra_ths += "<th>Description</th>"
    if has_user:
        extra_ths += "<th>User</th>"
    if has_host:
        extra_ths += "<th>Host</th>"

    rows = []
    for _, row in df.iterrows():
        score = float(row["anomaly_score"])
        is_anom = bool(row["is_anomaly"])
        t = _tier(score, is_anom)
        color = _score_color(score)
        ts = _fmt_ts(row["timestamp"])
        et = _e(_clean(row.get("event_type", "")))
        notes = _clean(row.get("anomaly_notes", ""))

        extra_tds = ""
        if has_desc:
            d = _clean(row.get("description", ""))
            extra_tds += (
                f'<td class="desc-cell" title="{_e(d)}">'
                f"{_e(d[:70] + '…' if len(d) > 70 else d)}</td>"
            )
        if has_user:
            extra_tds += f'<td>{_e(_clean(row.get("user", "")))}</td>'
        if has_host:
            extra_tds += f'<td>{_e(_clean(row.get("host", "")))}</td>'

        flag_attr = "true" if is_anom else "false"
        score_pct = f"{score * 100:.0f}%"
        notes_cell = (
            f'<td class="notes-cell">{_e(notes)}</td>'
            if notes
            else '<td class="notes-empty">—</td>'
        )

        rows.append(
            f'<tr class="row-{t}" data-flagged="{flag_attr}">'
            f'<td class="ts-cell mono">{ts}</td>'
            f"<td><span class=\"etype-badge\">{et}</span></td>"
            f"{extra_tds}"
            f'<td class="score-cell">'
            f'<div class="score-bar-wrap">'
            f'<div class="score-track"><div class="score-fill" style="width:{score_pct};background:{color}"></div></div>'
            f'<span class="score-num" style="color:{color}">{score:.3f}</span>'
            f"</div></td>"
            f"{notes_cell}"
            "</tr>"
        )

    total = len(df)
    return (
        '<div class="section">\n'
        '  <div class="section-title">Full Timeline</div>\n'
        '  <div class="controls">\n'
        '    <button class="btn" id="filter-btn" onclick="toggleFilter()">Flagged Only</button>\n'
        f'    <span class="result-count" id="row-count">Showing {total} of {total} events</span>\n'
        "  </div>\n"
        '  <div class="timeline-wrap">\n'
        "    <table>\n"
        "      <thead><tr>"
        "<th>Timestamp</th>"
        "<th>Event Type</th>"
        f"{extra_ths}"
        "<th>Score</th>"
        "<th>Anomaly Notes</th>"
        "</tr></thead>\n"
        "      <tbody>\n"
        + "\n".join(rows)
        + "\n      </tbody>\n    </table>\n  </div>\n</div>"
    )


def _footer(generated_at: str) -> str:
    return (
        '<footer class="page-footer">\n'
        '  <span class="footer-brand">thegr8val · Timeline Anomaly Highlighter</span>\n'
        f'  <span class="footer-meta">Generated {_e(generated_at)}</span>\n'
        "</footer>"
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_html_report(df: pd.DataFrame, title: str = "Timeline Anomaly Report") -> str:
    """
    Generate a self-contained HTML report from an analyzed timeline DataFrame.

    `df` must have been produced by `analyze()` (i.e. contain anomaly_score,
    is_anomaly, and anomaly_notes columns).

    Returns an HTML string ready to be written to a .html file.
    """
    required = {"timestamp", "event_type", "anomaly_score", "is_anomaly", "anomaly_notes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing columns: {missing}. Run analyze() first."
        )

    total = len(df)
    flagged = int(df["is_anomaly"].sum())
    flag_rate = flagged / total * 100 if total else 0.0
    max_score = float(df["anomaly_score"].max())
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    try:
        t0 = df["timestamp"].min().strftime("%Y-%m-%d")
        t1 = df["timestamp"].max().strftime("%Y-%m-%d")
        time_range = t0 if t0 == t1 else f"{t0} → {t1}"
    except Exception:
        time_range = "—"

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>{_e(title)}</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{_header(title, generated_at)}\n"
        '<div class="container">\n'
        f"{_summary_cards(total, flagged, flag_rate, max_score, time_range)}\n"
        f"{_score_strip(df)}\n"
        f"{_top_anomalies(df)}\n"
        f"{_table_section(df)}\n"
        "</div>\n"
        f"{_footer(generated_at)}\n"
        f"<script>{_JS}</script>\n"
        "</body>\n"
        "</html>"
    )
