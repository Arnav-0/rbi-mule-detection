"""Generate final submission with time windows."""
import pandas as pd
import logging
from src.temporal.window_detector import SuspiciousWindowDetector
from src.data.loader import load_transactions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load submission
sub = pd.read_csv('outputs/predictions/submission.csv')
logger.info("Submission shape: %s", sub.shape)

# Only detect windows for likely mules (save time)
likely_mules = sub[sub['is_mule'] > 0.3]['account_id'].values
logger.info("Detecting windows for %d likely mule accounts...", len(likely_mules))

# Load transactions
txn = load_transactions()

# Detect suspicious windows
detector = SuspiciousWindowDetector(z_threshold=2.0, min_window_days=7, extend_days=7)
windows = detector.detect_all(txn, likely_mules)
logger.info("Windows detected: %d", len(windows))

# Merge
sub['suspicious_start'] = ''
sub['suspicious_end'] = ''
if len(windows) > 0:
    windows_map = windows.set_index('account_id')
    for idx, row in sub.iterrows():
        if row['account_id'] in windows_map.index:
            w = windows_map.loc[row['account_id']]
            if isinstance(w, pd.DataFrame):
                w = w.iloc[0]
            sub.at[idx, 'suspicious_start'] = w.get('suspicious_start', '')
            sub.at[idx, 'suspicious_end'] = w.get('suspicious_end', '')

sub.to_csv('outputs/predictions/submission.csv', index=False)
logger.info("Final submission saved!")
has_windows = sub[sub['suspicious_start'] != '']
logger.info("Accounts with time windows: %d", len(has_windows))
print(has_windows.head(10))
