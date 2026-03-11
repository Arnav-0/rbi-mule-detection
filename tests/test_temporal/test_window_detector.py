"""Tests for SuspiciousWindowDetector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.temporal.window_detector import SuspiciousWindowDetector


def _make_txn(account_id: str, dates: list[str], amounts: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "account_id": account_id,
        "timestamp": pd.to_datetime(dates),
        "amount": amounts,
    })


class TestDetectClearAnomaly:
    def test_detect_clear_anomaly(self):
        """Steady low-volume account then large spike — window should be detected."""
        detector = SuspiciousWindowDetector(z_threshold=2.0, min_window_days=7, extend_days=3)

        # 120 days of baseline (~100 per day)
        base_dates = pd.date_range("2024-01-01", periods=120, freq="D")
        base_amounts = [100.0] * 120

        # 14-day spike (10x normal)
        spike_dates = pd.date_range("2024-05-01", periods=14, freq="D")
        spike_amounts = [1000.0] * 14

        all_dates = list(base_dates.strftime("%Y-%m-%d")) + list(spike_dates.strftime("%Y-%m-%d"))
        all_amounts = base_amounts + spike_amounts

        txn = _make_txn("ACC001", all_dates, all_amounts)
        result = detector.detect(txn, "ACC001")

        assert result is not None, "Should detect anomalous window"
        assert result["account_id"] == "ACC001"
        assert result["anomalous_days"] >= 7
        assert result["peak_z_score"] > 2.0
        start = pd.Timestamp(result["suspicious_start"])
        end = pd.Timestamp(result["suspicious_end"])
        assert start < end


class TestNoAnomaly:
    def test_no_anomaly(self):
        """Perfectly uniform transaction history — no window returned."""
        detector = SuspiciousWindowDetector(z_threshold=2.0, min_window_days=7)

        dates = pd.date_range("2024-01-01", periods=60, freq="D").strftime("%Y-%m-%d").tolist()
        amounts = [100.0] * 60
        txn = _make_txn("ACC002", dates, amounts)

        result = detector.detect(txn, "ACC002")
        assert result is None


class TestTemporalIoU:
    def test_iou_perfect(self):
        """Same window → IoU = 1.0."""
        iou = SuspiciousWindowDetector.compute_temporal_iou(
            "2024-03-01", "2024-03-31",
            "2024-03-01", "2024-03-31",
        )
        assert iou == pytest.approx(1.0, abs=1e-6)

    def test_iou_partial(self):
        """50% overlap → IoU ≈ 0.333 (15 intersect / 45 union for these windows)."""
        # pred: Jan 1 – Jan 30 (30 days), true: Jan 16 – Feb 14 (30 days)
        # intersection: Jan 16 – Jan 30 = 15 days
        # union: Jan 1 – Feb 14 = 45 days
        iou = SuspiciousWindowDetector.compute_temporal_iou(
            "2024-01-01", "2024-01-30",
            "2024-01-16", "2024-02-14",
        )
        assert iou == pytest.approx(15 / 45, abs=1e-6)

    def test_iou_no_overlap(self):
        """Non-overlapping windows → IoU = 0.0."""
        iou = SuspiciousWindowDetector.compute_temporal_iou(
            "2024-01-01", "2024-01-31",
            "2024-03-01", "2024-03-31",
        )
        assert iou == pytest.approx(0.0, abs=1e-6)

    def test_iou_partial_value(self):
        """Manual overlap check with explicit day counts."""
        # pred: Jan 1–10 (10 days), true: Jan 6–15 (10 days)
        # intersection: Jan 6–10 = 5 days, union: Jan 1–15 = 15 days
        iou = SuspiciousWindowDetector.compute_temporal_iou(
            "2024-01-01", "2024-01-10",
            "2024-01-06", "2024-01-15",
        )
        assert iou == pytest.approx(5 / 15, abs=1e-6)


class TestDetectAll:
    def test_detect_all_returns_dataframe(self):
        """detect_all returns a DataFrame with one row per account."""
        detector = SuspiciousWindowDetector(z_threshold=2.0, min_window_days=3, extend_days=1)

        base_dates = pd.date_range("2024-01-01", periods=60, freq="D")
        spike_dates = pd.date_range("2024-03-01", periods=10, freq="D")

        dates = list(base_dates.strftime("%Y-%m-%d")) + list(spike_dates.strftime("%Y-%m-%d"))
        amounts_mule = [50.0] * 60 + [5000.0] * 10
        amounts_clean = [100.0] * 70

        txn_mule = _make_txn("MULE01", dates, amounts_mule)
        txn_clean = _make_txn("CLEAN01", dates, amounts_clean)
        txn = pd.concat([txn_mule, txn_clean], ignore_index=True)

        result = detector.detect_all(txn, ["MULE01", "CLEAN01"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert set(result["account_id"]) == {"MULE01", "CLEAN01"}
