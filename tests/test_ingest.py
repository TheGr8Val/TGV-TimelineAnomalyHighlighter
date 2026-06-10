"""Tests for timeline_anomaly.ingest."""

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from timeline_anomaly.ingest import (
    TimelineConfig,
    autodetect_and_load,
    load_generic_csv,
    load_plaso_csv,
    NORMALIZED_COLS,
)


class TestLoadGenericCsv:
    def test_returns_normalized_columns(self, generic_csv_path):
        df = load_generic_csv(generic_csv_path)
        for col in NORMALIZED_COLS:
            assert col in df.columns

    def test_sorted_by_timestamp(self, generic_csv_path):
        df = load_generic_csv(generic_csv_path)
        assert df["timestamp"].is_monotonic_increasing

    def test_timestamp_is_utc(self, generic_csv_path):
        df = load_generic_csv(generic_csv_path)
        assert str(df["timestamp"].dt.tz) == "UTC"

    def test_row_count(self, generic_csv_path):
        df = load_generic_csv(generic_csv_path)
        assert len(df) == 8

    def test_custom_column_mapping(self, tmp_path):
        csv_content = "ts,etype,desc\n2024-01-01 10:00:00,LOGON,User logged in\n"
        p = tmp_path / "custom.csv"
        p.write_text(csv_content)
        config = TimelineConfig(timestamp_col="ts", event_type_col="etype", description_col="desc")
        df = load_generic_csv(p, config)
        assert len(df) == 1
        assert df["event_type"].iloc[0] == "LOGON"

    def test_bad_timestamps_dropped(self, tmp_path):
        csv_content = (
            "timestamp,event_type,description,user,host,filename\n"
            "NOT_A_DATE,LOGON,bad row,u,h,\n"
            "2024-01-01 09:00:00,LOGON,good row,u,h,\n"
        )
        p = tmp_path / "bad_ts.csv"
        p.write_text(csv_content)
        df = load_generic_csv(p)
        assert len(df) == 1
        assert df["event_type"].iloc[0] == "LOGON"

    def test_missing_optional_columns_filled(self, tmp_path):
        csv_content = "timestamp,event_type\n2024-01-01 09:00:00,LOGON\n"
        p = tmp_path / "minimal.csv"
        p.write_text(csv_content)
        df = load_generic_csv(p)
        assert df["user"].iloc[0] == ""
        assert df["host"].iloc[0] == ""


class TestLoadPlasoCsv:
    def test_returns_normalized_columns(self, plaso_csv_path):
        df = load_plaso_csv(plaso_csv_path)
        for col in NORMALIZED_COLS:
            assert col in df.columns

    def test_sorted_by_timestamp(self, plaso_csv_path):
        df = load_plaso_csv(plaso_csv_path)
        assert df["timestamp"].is_monotonic_increasing

    def test_row_count(self, plaso_csv_path):
        df = load_plaso_csv(plaso_csv_path)
        assert len(df) == 4

    def test_event_type_populated(self, plaso_csv_path):
        df = load_plaso_csv(plaso_csv_path)
        assert df["event_type"].notna().all()
        assert (df["event_type"] != "").all()


class TestAutodetect:
    def test_detects_generic(self, generic_csv_path):
        df = autodetect_and_load(generic_csv_path)
        assert len(df) == 8

    def test_detects_plaso(self, plaso_csv_path):
        df = autodetect_and_load(plaso_csv_path)
        assert len(df) == 4

    def test_plaso_markers_trigger_plaso_loader(self, tmp_path):
        # Minimal Plaso-like header
        content = (
            "date,time,timezone,MACB,source,sourcetype,type,user,host,"
            "short,desc,version,filename,inode,notes,format,extra\n"
            "03/15/2024,09:00:00,UTC,M...,EVT,Security,Logon,u,h,s,d,N/A,N/A,N/A,,evtx,\n"
        )
        p = tmp_path / "plaso_auto.csv"
        p.write_text(content)
        df = autodetect_and_load(p)
        assert df["event_type"].iloc[0] == "Logon"
