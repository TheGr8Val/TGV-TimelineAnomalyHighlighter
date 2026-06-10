"""Tests for timeline_anomaly.features."""

import numpy as np
import pandas as pd
import pytest

from timeline_anomaly.features import FEATURE_COLS, extract_features


class TestExtractFeatures:
    def test_output_shape(self, normal_df):
        feats = extract_features(normal_df)
        assert feats.shape[0] == len(normal_df)

    def test_all_feature_columns_present(self, normal_df):
        feats = extract_features(normal_df)
        for col in FEATURE_COLS:
            assert col in feats.columns, f"Missing feature: {col}"

    def test_no_nulls(self, normal_df):
        feats = extract_features(normal_df)
        assert not feats.isnull().any().any()

    def test_all_float64(self, normal_df):
        feats = extract_features(normal_df)
        assert (feats.dtypes == np.float64).all()

    def test_hour_range(self, normal_df):
        feats = extract_features(normal_df)
        assert feats["hour"].between(0, 23).all()

    def test_day_of_week_range(self, normal_df):
        feats = extract_features(normal_df)
        assert feats["day_of_week"].between(0, 6).all()

    def test_is_weekend_binary(self, normal_df):
        feats = extract_features(normal_df)
        assert set(feats["is_weekend"].unique()).issubset({0.0, 1.0})

    def test_inter_event_delta_non_negative(self, normal_df):
        feats = extract_features(normal_df)
        assert (feats["inter_event_delta_s"] >= 0).all()

    def test_first_event_delta_is_zero(self, normal_df):
        feats = extract_features(normal_df)
        # After sort, first event has no predecessor → delta = 0
        assert feats["inter_event_delta_s"].iloc[0] == 0.0

    def test_rarity_bounds(self, normal_df):
        feats = extract_features(normal_df)
        for col in ("event_type_rarity", "user_rarity", "host_rarity"):
            assert feats[col].between(0.0, 1.0).all(), f"{col} out of [0, 1]"

    def test_burst_rate_minimum_one(self, normal_df):
        feats = extract_features(normal_df)
        # Every event counts itself
        assert (feats["burst_rate_60s"] >= 1).all()

    def test_burst_rate_increases_during_burst(self, anomalous_df):
        feats = extract_features(anomalous_df)
        # The 3 AM cluster has events ~7-8 seconds apart; burst_rate should be > 1
        burst_mask = anomalous_df["event_type"] == "USB_WRITE"
        burst_rates = feats.loc[burst_mask, "burst_rate_60s"]
        # At least the second USB_WRITE should see count ≥ 2
        assert burst_rates.max() >= 2

    def test_weekend_flag_correct(self):
        # 2024-03-16 is a Saturday
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-03-16 10:00:00", tz="UTC")],
            "event_type": ["LOGON"],
            "user": ["u"],
            "host": ["h"],
        })
        feats = extract_features(df)
        assert feats["is_weekend"].iloc[0] == 1.0

    def test_workday_not_weekend(self):
        # 2024-03-11 is a Monday
        df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2024-03-11 09:00:00", tz="UTC")],
            "event_type": ["LOGON"],
            "user": ["u"],
            "host": ["h"],
        })
        feats = extract_features(df)
        assert feats["is_weekend"].iloc[0] == 0.0

    def test_rare_event_type_has_high_rarity(self, normal_df):
        feats = extract_features(normal_df)
        # LOGOFF appears once → should have higher rarity than LOGON or FILE_ACCESS
        event_types = normal_df["event_type"]
        logoff_idx = normal_df[event_types == "LOGOFF"].index
        file_idx = normal_df[event_types == "FILE_ACCESS"].index
        if len(logoff_idx) > 0 and len(file_idx) > 0:
            logoff_rarity = feats.loc[logoff_idx, "event_type_rarity"].mean()
            file_rarity = feats.loc[file_idx, "event_type_rarity"].mean()
            assert logoff_rarity >= file_rarity
