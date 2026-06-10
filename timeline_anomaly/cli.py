"""
CLI entry point for Timeline Anomaly Highlighter.

Usage:
    timeline-anomaly INPUT_CSV [OPTIONS]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from .detector import DetectorConfig
from .explain import generate_notes
from .features import extract_features
from .ingest import TimelineConfig, autodetect_and_load, load_generic_csv, load_plaso_csv
from .detector import AnomalyDetector
from .report import generate_html_report


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("input_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output-csv", "-o",
    default=None, metavar="PATH",
    help="Enriched CSV output path  [default: <input>_scored.csv]",
)
@click.option(
    "--output-html", "-r",
    default=None, metavar="PATH",
    help="HTML report output path  [default: <input>_report.html]",
)
@click.option(
    "--contamination", default=0.05, show_default=True, metavar="FLOAT",
    help="Expected fraction of anomalous events",
)
@click.option(
    "--threshold", default=0.65, show_default=True, metavar="FLOAT",
    help="Anomaly score cutoff (events ≥ threshold are flagged)",
)
@click.option(
    "--format", "fmt",
    type=click.Choice(["auto", "generic", "plaso"], case_sensitive=False),
    default="auto", show_default=True,
    help="Input format (auto-detected from header by default)",
)
@click.option("--no-html", is_flag=True, help="Skip HTML report")
@click.option("--no-csv",  is_flag=True, help="Skip enriched CSV output")
@click.option(
    "--title", default=None, metavar="TEXT",
    help="Title shown in the HTML report  [default: derived from filename]",
)
def cli(
    input_csv: Path,
    output_csv: Optional[str],
    output_html: Optional[str],
    contamination: float,
    threshold: float,
    fmt: str,
    no_html: bool,
    no_csv: bool,
    title: Optional[str],
) -> None:
    """Score a DFIR timeline CSV for anomalous events.

    Produces an enriched CSV (with anomaly_score, is_anomaly, anomaly_notes)
    and a self-contained HTML report with an annotated interactive timeline.

    \b
    Examples:
      timeline-anomaly timeline.csv
      timeline-anomaly timeline.csv --threshold 0.70 --no-csv
      timeline-anomaly timeline.csv --format plaso -o out.csv -r report.html
    """
    _banner()

    # ── 1. Load ──────────────────────────────────────────────────────────────
    click.echo(_step(1, 3, "Loading timeline …"))
    try:
        if fmt == "generic":
            df = load_generic_csv(input_csv)
            fmt_label = "generic CSV"
        elif fmt == "plaso":
            df = load_plaso_csv(input_csv)
            fmt_label = "Plaso l2t_csv"
        else:
            df = autodetect_and_load(input_csv)
            fmt_label = "Plaso l2t_csv" if "plaso" in _sniff_format(input_csv) else "generic CSV"
    except Exception as exc:
        raise click.ClickException(f"Failed to load {input_csv}: {exc}") from exc

    click.echo(f"     {len(df):,} events  ·  format: {fmt_label}")

    # ── 2. Score ─────────────────────────────────────────────────────────────
    click.echo(_step(2, 3, "Extracting features + scoring …"))
    try:
        feats = extract_features(df)
        det = AnomalyDetector(DetectorConfig(contamination=contamination))
        scores = det.fit_score(feats)
        flags = det.flag(threshold)
        notes = generate_notes(feats)
    except Exception as exc:
        raise click.ClickException(f"Scoring failed: {exc}") from exc

    df["anomaly_score"] = scores.round(4)
    df["is_anomaly"] = flags
    df["anomaly_notes"] = notes

    flagged = int(flags.sum())
    flag_rate = flagged / len(df) * 100
    max_score = float(scores.max())

    # ── 3. Write outputs ─────────────────────────────────────────────────────
    click.echo(_step(3, 3, "Writing outputs …"))

    stem = input_csv.stem
    parent = input_csv.parent

    csv_path = Path(output_csv) if output_csv else parent / f"{stem}_scored.csv"
    html_path = Path(output_html) if output_html else parent / f"{stem}_report.html"

    if not no_csv:
        df.to_csv(csv_path, index=False)

    if not no_html:
        report_title = title or f"Timeline Anomaly Report — {input_csv.name}"
        html_content = generate_html_report(df, title=report_title)
        html_path.write_text(html_content, encoding="utf-8")

    # ── Summary ──────────────────────────────────────────────────────────────
    _print_summary(len(df), flagged, flag_rate, max_score, df)

    click.echo("")
    if not no_csv:
        click.echo(f"  → CSV   {csv_path}")
    if not no_html:
        click.echo(f"  → HTML  {html_path}")
    click.echo("")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _banner() -> None:
    click.echo("")
    click.echo(click.style("  Timeline Anomaly Highlighter", fg="magenta", bold=True)
               + click.style(" · thegr8val", fg="bright_black"))
    click.echo(click.style("  " + "─" * 42, fg="bright_black"))
    click.echo("")


def _step(n: int, total: int, msg: str) -> str:
    return click.style(f"  [{n}/{total}]", fg="cyan") + f" {msg}"


def _sniff_format(path: Path) -> str:
    try:
        header = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        markers = {"MACB", "sourcetype", "inode"}
        cols = set(c.strip() for c in header.split(","))
        return "plaso" if len(markers & cols) >= 2 else "generic"
    except Exception:
        return "generic"


def _score_bar(score: float, width: int = 20) -> str:
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    if score >= 0.8:
        color = "red"
    elif score >= 0.65:
        color = "yellow"
    else:
        color = "bright_black"
    return click.style(bar, fg=color)


def _print_summary(
    total: int, flagged: int, flag_rate: float, max_score: float, df
) -> None:
    click.echo("")
    w = 46
    click.echo("  " + click.style("┌─ Results " + "─" * (w - 10) + "┐", fg="bright_black"))

    def _row(label: str, value: str) -> None:
        pad = w - len(label) - len(value) - 2
        click.echo(
            "  "
            + click.style("│", fg="bright_black")
            + f"  {label}"
            + " " * pad
            + f"{value}  "
            + click.style("│", fg="bright_black")
        )

    _row("Total events ", f"{total:,}")
    flagged_str = click.style(f"{flagged:,}", fg="red" if flagged else "green") + f"  ({flag_rate:.1f}%)"
    _row("Flagged      ", f"{flagged}  ({flag_rate:.1f}%)")
    _row("Max score    ", f"{max_score:.4f}")
    click.echo("  " + click.style("└" + "─" * (w + 2) + "┘", fg="bright_black"))

    if flagged == 0:
        click.echo("")
        click.echo(click.style("  No events flagged.", fg="green"))
        return

    top = df[df["is_anomaly"]].nlargest(3, "anomaly_score")
    click.echo("")
    click.echo(click.style("  Top flagged events:", bold=True))
    click.echo("")
    for _, row in top.iterrows():
        score = float(row["anomaly_score"])
        ts = str(row["timestamp"])[:19] if not hasattr(row["timestamp"], "strftime") \
            else row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        et = str(row.get("event_type", ""))
        notes = str(row.get("anomaly_notes", ""))
        score_col = "red" if score >= 0.8 else "yellow"
        click.echo(
            "  "
            + click.style(f"[{score:.3f}]", fg=score_col, bold=True)
            + "  "
            + click.style(ts, fg="bright_black")
            + "  "
            + click.style(et, bold=True)
        )
        if notes and notes not in ("nan", ""):
            # Wrap notes to 60 chars
            click.echo(click.style(f"           {notes[:80]}", fg="bright_black"))
        click.echo("")
