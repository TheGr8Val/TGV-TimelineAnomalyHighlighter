"""Tests for timeline_anomaly.report."""

import pytest
import pandas as pd

from timeline_anomaly.report import generate_html_report


def _make_scored_df() -> pd.DataFrame:
    """Minimal analyzed DataFrame with all required columns."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-03-11 09:00:00",
            "2024-03-11 10:00:00",
            "2024-03-12 03:00:00",
        ], utc=True),
        "event_type": ["LOGON", "FILE_ACCESS", "USB_WRITE"],
        "description": ["User logon", "Read file", "Copied to USB"],
        "user": ["jsmith", "jsmith", "jsmith"],
        "host": ["WS01", "WS01", "WS01"],
        "filename": ["", "C:\\doc.docx", "E:\\data.zip"],
        "anomaly_score": [0.12, 0.18, 0.91],
        "is_anomaly": [False, False, True],
        "anomaly_notes": [
            "",
            "",
            "Unusual hour (3:00; typical 9–17h); Rare event type (1.0%)",
        ],
    })
    return df


class TestGenerateHtmlReport:
    def test_returns_string(self):
        html = generate_html_report(_make_scored_df())
        assert isinstance(html, str)

    def test_is_valid_html_structure(self):
        html = generate_html_report(_make_scored_df())
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body>" in html

    def test_contains_title(self):
        html = generate_html_report(_make_scored_df(), title="My DFIR Report")
        assert "My DFIR Report" in html

    def test_default_title(self):
        html = generate_html_report(_make_scored_df())
        assert "Timeline Anomaly Report" in html

    def test_contains_brand(self):
        html = generate_html_report(_make_scored_df())
        assert "thegr8val" in html

    def test_contains_all_event_types(self):
        html = generate_html_report(_make_scored_df())
        assert "LOGON" in html
        assert "FILE_ACCESS" in html
        assert "USB_WRITE" in html

    def test_flagged_row_marked(self):
        html = generate_html_report(_make_scored_df())
        assert 'data-flagged="true"' in html
        assert 'data-flagged="false"' in html

    def test_flagged_event_in_top_anomalies(self):
        html = generate_html_report(_make_scored_df())
        # The USB_WRITE event (score 0.91) should appear in the top anomalies section
        assert "0.910" in html

    def test_score_strip_present(self):
        html = generate_html_report(_make_scored_df())
        assert "score-strip" in html
        assert "strip-bar" in html

    def test_filter_button_present(self):
        html = generate_html_report(_make_scored_df())
        assert "Flagged Only" in html
        assert "toggleFilter" in html

    def test_anomaly_notes_in_output(self):
        html = generate_html_report(_make_scored_df())
        assert "Unusual hour" in html

    def test_summary_stats_present(self):
        html = generate_html_report(_make_scored_df())
        # Total events = 3, flagged = 1
        assert "Total Events" in html
        assert "Flagged" in html
        assert "Max Score" in html

    def test_raises_on_missing_columns(self):
        bad_df = pd.DataFrame({"timestamp": [], "event_type": []})
        with pytest.raises(ValueError, match="missing columns"):
            generate_html_report(bad_df)

    def test_no_anomalies_omits_top_section(self):
        df = _make_scored_df()
        df["is_anomaly"] = False
        html = generate_html_report(df)
        # Top Flagged Events section should not appear
        assert "Top Flagged Events" not in html

    def test_html_is_self_contained(self):
        html = generate_html_report(_make_scored_df())
        # No external CDN or stylesheet links
        assert "cdn.jsdelivr.net" not in html
        assert "unpkg.com" not in html
        assert 'rel="stylesheet"' not in html

    def test_css_uses_tgv_brand_colors(self):
        html = generate_html_report(_make_scored_df())
        assert "#A855F7" in html
        assert "#EC4899" in html
        assert "#0A0A12" in html
