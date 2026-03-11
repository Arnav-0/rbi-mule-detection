import numpy as np
import pandas as pd
from datetime import timedelta


class SuspiciousWindowDetector:
    def __init__(self, z_threshold=2.0, min_window_days=7, extend_days=7):
        self.z_threshold = z_threshold
        self.min_window_days = min_window_days
        self.extend_days = extend_days

    def detect(self, txn, account_id):
        # Filter transactions for this account
        acct_txn = txn[txn['account_id'] == account_id].copy()
        if len(acct_txn) == 0:
            return None

        # Support multiple column name conventions
        date_col = next((c for c in ['transaction_date', 'date', 'timestamp'] if c in acct_txn.columns), None)
        amt_col = next((c for c in ['transaction_amount', 'amount'] if c in acct_txn.columns), None)
        if date_col is None or amt_col is None:
            return None
        acct_txn['date'] = pd.to_datetime(acct_txn[date_col]).dt.date

        # Daily aggregates
        daily = acct_txn.groupby('date').agg(
            total_amount=(amt_col, 'sum'),
            txn_count=(amt_col, 'count')
        ).reset_index()
        daily['date'] = pd.to_datetime(daily['date'])

        # Fill missing days with 0
        date_range = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
        daily = daily.set_index('date').reindex(date_range, fill_value=0).reset_index()
        daily.rename(columns={'index': 'date'}, inplace=True)

        if len(daily) < 14:  # Need enough data for rolling stats
            return None

        # 90-day rolling mean and std of daily volume
        window = min(90, len(daily) - 1)
        daily['rolling_mean'] = daily['total_amount'].rolling(window=window, min_periods=7).mean()
        daily['rolling_std'] = daily['total_amount'].rolling(window=window, min_periods=7).std()

        # Z-score each day
        daily['z_score'] = np.where(
            daily['rolling_std'] > 0,
            (daily['total_amount'] - daily['rolling_mean']) / daily['rolling_std'],
            0
        )

        # Find contiguous periods where z_score > threshold
        daily['anomalous'] = daily['z_score'] > self.z_threshold
        daily['group'] = (~daily['anomalous']).cumsum()

        anomalous_periods = []
        for group_id, group_df in daily[daily['anomalous']].groupby('group'):
            if len(group_df) >= 1:  # At least 1 anomalous day
                anomalous_periods.append({
                    'start': group_df['date'].min(),
                    'end': group_df['date'].max(),
                    'days': len(group_df),
                    'peak_z': group_df['z_score'].max(),
                })

        if not anomalous_periods:
            return None

        # Select longest anomalous period
        best = max(anomalous_periods, key=lambda x: x['days'])

        # Extend by extend_days each side, clip to actual date range
        actual_min = daily['date'].min()
        actual_max = daily['date'].max()

        suspicious_start = max(best['start'] - timedelta(days=self.extend_days), actual_min)
        suspicious_end = min(best['end'] + timedelta(days=self.extend_days), actual_max)

        # Ensure minimum window
        window_days = (suspicious_end - suspicious_start).days
        if window_days < self.min_window_days:
            extra = self.min_window_days - window_days
            suspicious_start = max(suspicious_start - timedelta(days=extra // 2), actual_min)
            suspicious_end = min(suspicious_end + timedelta(days=(extra + 1) // 2), actual_max)

        return {
            'account_id': account_id,
            'suspicious_start': suspicious_start.isoformat(),
            'suspicious_end': suspicious_end.isoformat(),
            'peak_z_score': float(best['peak_z']),
            'anomalous_days': int(best['days']),
        }

    def detect_all(self, txn, account_ids):
        # Pre-filter transactions for target accounts for speed
        target_set = set(account_ids)
        txn_filtered = txn[txn['account_id'].isin(target_set)]
        results = []
        total = len(account_ids)
        for i, acct_id in enumerate(account_ids):
            if i % 500 == 0:
                print(f"  Window detection: {i}/{total}")
            result = self.detect(txn_filtered, acct_id)
            if result is not None:
                results.append(result)
        print(f"  Window detection: {total}/{total} done")
        if not results:
            return pd.DataFrame(columns=['account_id', 'suspicious_start', 'suspicious_end', 'peak_z_score', 'anomalous_days'])
        return pd.DataFrame(results)

    @staticmethod
    def compute_temporal_iou(pred_start, pred_end, true_start, true_end):
        pred_start = pd.Timestamp(pred_start)
        pred_end = pd.Timestamp(pred_end)
        true_start = pd.Timestamp(true_start)
        true_end = pd.Timestamp(true_end)

        intersection_start = max(pred_start, true_start)
        intersection_end = min(pred_end, true_end)

        if intersection_start >= intersection_end:
            return 0.0

        intersection = (intersection_end - intersection_start).total_seconds()

        pred_duration = (pred_end - pred_start).total_seconds()
        true_duration = (true_end - true_start).total_seconds()
        union = pred_duration + true_duration - intersection

        if union <= 0:
            return 0.0

        return intersection / union
