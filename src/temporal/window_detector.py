"""Suspicious time window detector using z-score anomaly detection."""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SuspiciousWindowDetector:
    def __init__(
        self,
        z_threshold: float = 2.0,
        min_window_days: int = 7,
        extend_days: int = 7,
        rolling_window: int = 90,
    ):
        self.z_threshold = z_threshold
        self.min_window_days = min_window_days
        self.extend_days = extend_days
        self.rolling_window = rolling_window

    def detect(self, txn: pd.DataFrame, account_id: str) -> Optional[dict]:
        """Detect the most anomalous contiguous time window for one account.

        Returns dict with suspicious_start, suspicious_end, peak_z_score, anomalous_days.
        Returns None if no anomalous window found.
        """
        acc_txn = txn[txn["account_id"] == account_id].copy()
        if acc_txn.empty:
            return None

        acc_txn["date"] = pd.to_datetime(acc_txn["timestamp"]).dt.normalize()
        daily = acc_txn.groupby("date")["amount"].sum().rename("daily_amount")

        date_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(date_range, fill_value=0.0)

        if len(daily) < self.rolling_window + 1:
            rolling_mean = daily.expanding(min_periods=1).mean()
            rolling_std = daily.expanding(min_periods=1).std().fillna(1.0)
        else:
            rolling_mean = daily.rolling(self.rolling_window, min_periods=1).mean()
            rolling_std = daily.rolling(self.rolling_window, min_periods=1).std().fillna(1.0)

        rolling_std = rolling_std.replace(0.0, 1.0)
        z_scores = (daily - rolling_mean) / rolling_std

        above = z_scores >= self.z_threshold
        if not above.any():
            return None

        # Find contiguous anomalous periods
        periods = []
        in_period = False
        start_idx = None
        for i, (date, flag) in enumerate(above.items()):
            if flag and not in_period:
                in_period = True
                start_idx = i
            elif not flag and in_period:
                in_period = False
                periods.append((start_idx, i - 1))
        if in_period:
            periods.append((start_idx, len(above) - 1))

        if not periods:
            return None

        # Select period with highest total anomalous days × peak z-score
        best_period = max(
            periods,
            key=lambda p: (p[1] - p[0] + 1) * z_scores.iloc[p[0]:p[1] + 1].max(),
        )

        dates = daily.index
        raw_start = dates[best_period[0]]
        raw_end = dates[best_period[1]]
        peak_z = float(z_scores.iloc[best_period[0]:best_period[1] + 1].max())
        anomalous_days = best_period[1] - best_period[0] + 1

        if anomalous_days < self.min_window_days:
            return None

        # Extend window by extend_days each side, clip to actual date range
        ext_start = max(raw_start - pd.Timedelta(days=self.extend_days), dates[0])
        ext_end = min(raw_end + pd.Timedelta(days=self.extend_days), dates[-1])

        return {
            "account_id": account_id,
            "suspicious_start": ext_start.isoformat(),
            "suspicious_end": ext_end.isoformat(),
            "peak_z_score": round(peak_z, 4),
            "anomalous_days": anomalous_days,
        }

    def detect_all(self, txn: pd.DataFrame, account_ids: list[str]) -> pd.DataFrame:
        """Batch detection across multiple accounts."""
        rows = []
        for account_id in account_ids:
            result = self.detect(txn, account_id)
            if result is not None:
                rows.append(result)
            else:
                rows.append({
                    "account_id": account_id,
                    "suspicious_start": None,
                    "suspicious_end": None,
                    "peak_z_score": None,
                    "anomalous_days": 0,
                })
        return pd.DataFrame(rows)

    @staticmethod
    def compute_temporal_iou(
        pred_start: str,
        pred_end: str,
        true_start: str,
        true_end: str,
    ) -> float:
        """Compute Intersection over Union for two time windows (ISO date strings)."""
        ps = pd.Timestamp(pred_start)
        pe = pd.Timestamp(pred_end)
        ts = pd.Timestamp(true_start)
        te = pd.Timestamp(true_end)

        intersection_start = max(ps, ts)
        intersection_end = min(pe, te)

        if intersection_end < intersection_start:
            return 0.0

        intersection_days = (intersection_end - intersection_start).days + 1
        union_start = min(ps, ts)
        union_end = max(pe, te)
        union_days = (union_end - union_start).days + 1

        return intersection_days / union_days if union_days > 0 else 0.0
