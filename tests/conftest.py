"""Shared fixtures for timeline_anomaly tests."""

import io
import textwrap
from pathlib import Path

import pandas as pd
import pytest


GENERIC_CSV = textwrap.dedent("""\
    timestamp,event_type,description,user,host,filename
    2024-03-11 09:00:00,LOGON,Logon,jsmith,WS01,
    2024-03-11 09:05:00,FILE_ACCESS,Read doc,jsmith,WS01,C:\\doc.docx
    2024-03-11 09:10:00,FILE_ACCESS,Read doc,jsmith,WS01,C:\\report.docx
    2024-03-11 09:20:00,FILE_WRITE,Saved doc,jsmith,WS01,C:\\report_v2.docx
    2024-03-11 10:00:00,PROCESS_EXEC,Launched app,jsmith,WS01,C:\\app.exe
    2024-03-11 10:30:00,FILE_ACCESS,Read file,jsmith,WS01,\\\\FS01\\share
    2024-03-11 11:00:00,FILE_ACCESS,Read file,jsmith,WS01,\\\\FS01\\share\\data.xlsx
    2024-03-11 16:00:00,LOGOFF,Logoff,jsmith,WS01,
""")

PLASO_CSV = textwrap.dedent("""\
    date,time,timezone,MACB,source,sourcetype,type,user,host,short,desc,version,filename,inode,notes,format,extra
    03/11/2024,09:00:00,UTC,M...,EVT,Security,Logon,jsmith,WS01,Logon,Interactive logon,N/A,N/A,N/A,,evtx,
    03/11/2024,09:05:00,UTC,.A..,FILE,OS:FS,Last access time,jsmith,WS01,doc.docx,File read,N/A,C:\\doc.docx,N/A,,filestat,
    03/11/2024,10:00:00,UTC,MACB,FILE,OS:FS,Creation time,jsmith,WS01,app.exe,Process exec,N/A,C:\\app.exe,N/A,,filestat,
    03/11/2024,16:00:00,UTC,M...,EVT,Security,Logoff,jsmith,WS01,Logoff,Logoff event,N/A,N/A,N/A,,evtx,
""")


@pytest.fixture()
def generic_csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "generic.csv"
    p.write_text(GENERIC_CSV)
    return p


@pytest.fixture()
def plaso_csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "plaso.csv"
    p.write_text(PLASO_CSV)
    return p


@pytest.fixture()
def normal_df() -> pd.DataFrame:
    """Normalized timeline with 9-5 workday events (no anomalies)."""
    from timeline_anomaly.ingest import load_generic_csv
    return load_generic_csv(io.StringIO(GENERIC_CSV))


@pytest.fixture()
def anomalous_df(normal_df: pd.DataFrame) -> pd.DataFrame:
    """Normal timeline plus obvious 3 AM burst — should score high."""
    import numpy as np
    from timeline_anomaly.ingest import NORMALIZED_COLS

    burst_rows = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2024-03-12 03:11:00", tz="UTC"),
            "event_type": "USB_WRITE",
            "description": "Copied to USB",
            "user": "jsmith",
            "host": "WS01",
            "filename": "E:\\secret.zip",
        },
        {
            "timestamp": pd.Timestamp("2024-03-12 03:11:08", tz="UTC"),
            "event_type": "USB_WRITE",
            "description": "Copied to USB",
            "user": "jsmith",
            "host": "WS01",
            "filename": "E:\\secret2.zip",
        },
        {
            "timestamp": pd.Timestamp("2024-03-12 03:11:15", tz="UTC"),
            "event_type": "REGISTRY_WRITE",
            "description": "Modified run key",
            "user": "jsmith",
            "host": "WS01",
            "filename": "HKCU\\Run",
        },
    ])
    combined = pd.concat([normal_df, burst_rows], ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)
