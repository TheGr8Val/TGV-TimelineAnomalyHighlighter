"""Tests for timeline_anomaly.cli."""

import textwrap
from pathlib import Path

import pytest
from click.testing import CliRunner

from timeline_anomaly.cli import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def generic_path(tmp_path: Path) -> Path:
    content = textwrap.dedent("""\
        timestamp,event_type,description,user,host,filename
        2024-03-11 09:00:00,LOGON,Logon,jsmith,WS01,
        2024-03-11 09:05:00,FILE_ACCESS,Read doc,jsmith,WS01,C:\\doc.docx
        2024-03-11 09:10:00,FILE_ACCESS,Read doc,jsmith,WS01,C:\\report.docx
        2024-03-11 09:20:00,FILE_WRITE,Saved doc,jsmith,WS01,C:\\report_v2.docx
        2024-03-11 10:00:00,PROCESS_EXEC,Launched app,jsmith,WS01,C:\\app.exe
        2024-03-11 10:30:00,FILE_ACCESS,Read file,jsmith,WS01,\\\\FS01\\share
        2024-03-11 11:00:00,FILE_ACCESS,Read file,jsmith,WS01,\\\\FS01\\data.xlsx
        2024-03-11 16:00:00,LOGOFF,Logoff,jsmith,WS01,
        2024-03-11 16:30:00,LOGON,Logon,jsmith,WS01,
        2024-03-11 16:45:00,FILE_ACCESS,Read doc,jsmith,WS01,C:\\report.docx
        2024-03-12 03:00:00,USB_WRITE,Copied to USB,jsmith,WS01,E:\\data.zip
        2024-03-12 03:00:08,USB_WRITE,Copied to USB,jsmith,WS01,E:\\data2.zip
    """)
    p = tmp_path / "test_timeline.csv"
    p.write_text(content)
    return p


class TestCliScore:
    def test_exits_zero(self, runner, generic_path, tmp_path):
        result = runner.invoke(cli, [str(generic_path), "--no-html", "--no-csv"])
        assert result.exit_code == 0, result.output

    def test_shows_event_count(self, runner, generic_path):
        result = runner.invoke(cli, [str(generic_path), "--no-html", "--no-csv"])
        assert "12" in result.output or "events" in result.output

    def test_writes_csv(self, runner, generic_path, tmp_path):
        out_csv = tmp_path / "out.csv"
        result = runner.invoke(cli, [str(generic_path), "--no-html", "-o", str(out_csv)])
        assert result.exit_code == 0, result.output
        assert out_csv.exists()

    def test_csv_has_required_columns(self, runner, generic_path, tmp_path):
        import pandas as pd
        out_csv = tmp_path / "out.csv"
        runner.invoke(cli, [str(generic_path), "--no-html", "-o", str(out_csv)])
        df = pd.read_csv(out_csv)
        for col in ("anomaly_score", "is_anomaly", "anomaly_notes"):
            assert col in df.columns

    def test_writes_html(self, runner, generic_path, tmp_path):
        out_html = tmp_path / "report.html"
        result = runner.invoke(cli, [str(generic_path), "--no-csv", "-r", str(out_html)])
        assert result.exit_code == 0, result.output
        assert out_html.exists()
        content = out_html.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "thegr8val" in content

    def test_html_is_utf8(self, runner, generic_path, tmp_path):
        out_html = tmp_path / "report.html"
        runner.invoke(cli, [str(generic_path), "--no-csv", "-r", str(out_html)])
        content = out_html.read_bytes()
        content.decode("utf-8")  # should not raise

    def test_no_html_skips_report(self, runner, generic_path, tmp_path):
        out_html = tmp_path / "should_not_exist.html"
        runner.invoke(cli, [str(generic_path), "--no-html", "--no-csv", "-r", str(out_html)])
        assert not out_html.exists()

    def test_no_csv_skips_csv(self, runner, generic_path, tmp_path):
        out_csv = tmp_path / "should_not_exist.csv"
        runner.invoke(cli, [str(generic_path), "--no-csv", "--no-html", "-o", str(out_csv)])
        assert not out_csv.exists()

    def test_custom_threshold(self, runner, generic_path, tmp_path):
        out_csv = tmp_path / "out.csv"
        result = runner.invoke(
            cli, [str(generic_path), "--threshold", "0.99", "--no-html", "-o", str(out_csv)]
        )
        assert result.exit_code == 0, result.output
        import pandas as pd
        df = pd.read_csv(out_csv)
        # With threshold 0.99, very few events should be flagged
        assert df["is_anomaly"].sum() <= len(df)

    def test_nonexistent_file_fails(self, runner, tmp_path):
        result = runner.invoke(cli, [str(tmp_path / "ghost.csv"), "--no-html", "--no-csv"])
        assert result.exit_code != 0

    def test_summary_in_output(self, runner, generic_path):
        result = runner.invoke(cli, [str(generic_path), "--no-html", "--no-csv"])
        assert "Results" in result.output
        assert "Flagged" in result.output

    def test_explicit_format_generic(self, runner, generic_path, tmp_path):
        out_csv = tmp_path / "out.csv"
        result = runner.invoke(
            cli, [str(generic_path), "--format", "generic", "--no-html", "-o", str(out_csv)]
        )
        assert result.exit_code == 0, result.output

    def test_custom_title_in_html(self, runner, generic_path, tmp_path):
        out_html = tmp_path / "report.html"
        runner.invoke(
            cli,
            [str(generic_path), "--no-csv", "-r", str(out_html), "--title", "My Custom Title"],
        )
        content = out_html.read_text(encoding="utf-8")
        assert "My Custom Title" in content

    def test_help_exits_zero(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "DFIR" in result.output or "timeline" in result.output.lower()
