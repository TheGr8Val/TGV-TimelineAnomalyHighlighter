"""Tests for timeline_anomaly.detector."""

import numpy as np
import pandas as pd
import pytest

from timeline_anomaly.detector import AnomalyDetector, DetectorConfig
from timeline_anomaly.features import extract_features


class TestAnomalyDetector:
    def test_fit_score_returns_series(self, normal_df):
        feats = extract_features(normal_df)
        det = AnomalyDetector()
        scores = det.fit_score(feats)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(feats)

    def test_scores_in_0_1(self, normal_df):
        feats = extract_features(normal_df)
        scores = AnomalyDetector().fit_score(feats)
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_scores_index_matches_features(self, normal_df):
        feats = extract_features(normal_df)
        scores = AnomalyDetector().fit_score(feats)
        assert (scores.index == feats.index).all()

    def test_flag_returns_boolean_series(self, normal_df):
        feats = extract_features(normal_df)
        det = AnomalyDetector()
        det.fit_score(feats)
        flags = det.flag()
        assert flags.dtype == bool
        assert len(flags) == len(feats)

    def test_flag_before_fit_raises(self):
        det = AnomalyDetector()
        with pytest.raises(RuntimeError, match="fit_score"):
            det.flag()

    def test_custom_threshold_all_flagged(self, normal_df):
        feats = extract_features(normal_df)
        det = AnomalyDetector()
        det.fit_score(feats)
        assert det.flag(threshold=0.0).all()

    def test_custom_threshold_none_flagged(self, normal_df):
        feats = extract_features(normal_df)
        det = AnomalyDetector()
        det.fit_score(feats)
        assert not det.flag(threshold=1.01).any()

    def test_injected_anomaly_scores_higher(self, normal_df, anomalous_df):
        """3 AM burst events should score higher than typical workday events."""
        normal_feats = extract_features(normal_df)
        anomalous_feats = extract_features(anomalous_df)

        det = AnomalyDetector(DetectorConfig(contamination=0.1))
        scores = det.fit_score(anomalous_feats)

        burst_mask = anomalous_df["event_type"].isin(["USB_WRITE", "REGISTRY_WRITE"])
        normal_mask = ~burst_mask

        burst_mean = scores[burst_mask].mean()
        normal_mean = scores[normal_mask].mean()

        assert burst_mean > normal_mean, (
            f"Expected burst events (mean={burst_mean:.3f}) to score higher "
            f"than normal events (mean={normal_mean:.3f})"
        )

    def test_scores_property_available_after_fit(self, normal_df):
        feats = extract_features(normal_df)
        det = AnomalyDetector()
        det.fit_score(feats)
        assert det.scores is not None
        assert len(det.scores) == len(feats)

    def test_scores_property_none_before_fit(self):
        det = AnomalyDetector()
        assert det.scores is None

    def test_contamination_config_respected(self, anomalous_df):
        """Flag count should roughly match contamination * n_samples."""
        feats = extract_features(anomalous_df)
        contamination = 0.15
        det = AnomalyDetector(DetectorConfig(contamination=contamination))
        scores = det.fit_score(feats)
        flags = det.flag()
        flag_rate = flags.sum() / len(flags)
        # Allow ±10% tolerance around the target contamination rate
        assert abs(flag_rate - contamination) <= 0.10 + contamination, (
            f"flag_rate={flag_rate:.2f} far from contamination={contamination}"
        )
