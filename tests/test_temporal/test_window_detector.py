import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.temporal.window_detector import SuspiciousWindowDetector


@pytest.fixture
def detector():
    return SuspiciousWindowDetector(z_threshold=2.0, min_window_days=7, extend_days=7)


def _make_txn(account_id, dates, amounts):
    return pd.DataFrame({
        'account_id': [account_id] * len(dates),
        'date': dates,
        'amount': amounts,
    })


class TestDetectClearAnomaly:
    def test_detect_clear_anomaly(self, detector):
        # 90 days of steady activity, then 10 days of spike
        dates = []
        amounts = []
        base_date = datetime(2024, 1, 1)
        np.random.seed(42)

        # Normal period: 90 days, ~100 per day
        for i in range(90):
            d = base_date + timedelta(days=i)
            dates.append(d)
            amounts.append(np.random.normal(100, 10))

        # Spike period: 10 days, ~1000 per day
        for i in range(90, 100):
            d = base_date + timedelta(days=i)
            dates.append(d)
            amounts.append(np.random.normal(1000, 50))

        # More normal: 30 days
        for i in range(100, 130):
            d = base_date + timedelta(days=i)
            dates.append(d)
            amounts.append(np.random.normal(100, 10))

        txn = _make_txn('ACCT001', dates, amounts)
        result = detector.detect(txn, 'ACCT001')

        assert result is not None
        assert result['account_id'] == 'ACCT001'
        assert result['peak_z_score'] > 2.0
        assert result['anomalous_days'] >= 1


class TestNoAnomaly:
    def test_no_anomaly(self, detector):
        # Uniform activity for 120 days
        dates = []
        amounts = []
        base_date = datetime(2024, 1, 1)
        np.random.seed(42)

        for i in range(120):
            d = base_date + timedelta(days=i)
            dates.append(d)
            amounts.append(100.0)  # Exactly the same every day

        txn = _make_txn('ACCT002', dates, amounts)
        result = detector.detect(txn, 'ACCT002')

        # Uniform data should have zero std -> z_score = 0 -> no anomaly
        assert result is None


class TestTemporalIoU:
    def test_iou_perfect(self):
        iou = SuspiciousWindowDetector.compute_temporal_iou(
            '2024-01-01', '2024-01-31',
            '2024-01-01', '2024-01-31'
        )
        assert iou == pytest.approx(1.0)

    def test_iou_partial(self):
        # pred: Jan 1 - Jan 20, true: Jan 11 - Jan 30
        # intersection: Jan 11 - Jan 20 = 9 days
        # union: 19 + 19 - 9 = 29 days
        iou = SuspiciousWindowDetector.compute_temporal_iou(
            '2024-01-01', '2024-01-20',
            '2024-01-11', '2024-01-30'
        )
        assert 0.0 < iou < 1.0
        expected = 9.0 / 29.0
        assert iou == pytest.approx(expected, rel=0.01)

    def test_iou_no_overlap(self):
        iou = SuspiciousWindowDetector.compute_temporal_iou(
            '2024-01-01', '2024-01-10',
            '2024-02-01', '2024-02-10'
        )
        assert iou == 0.0
