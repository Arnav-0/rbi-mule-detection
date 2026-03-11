"""Tests for src/utils/metrics.py and src/utils/registry helpers."""
import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import (
    compute_all_metrics,
    find_f1_threshold,
    find_optimal_threshold,
    temporal_iou,
)
from src.features.registry import (
    FEATURE_REGISTRY,
    get_all_feature_names,
    get_features_by_group,
    get_high_power_features,
    print_feature_summary,
)
from src.utils.logging_config import setup_logger
from src.data.preprocessor import preprocess_profile


# ── metrics ──────────────────────────────────────────────────────────────────

@pytest.fixture
def binary_labels():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    return y_true, y_prob


def test_compute_all_metrics_keys(binary_labels):
    y_true, y_prob = binary_labels
    result = compute_all_metrics(y_true, y_prob)
    expected_keys = {"auc_roc", "auc_pr", "f1", "precision", "recall",
                     "brier_score", "confusion_matrix", "threshold_used"}
    assert set(result.keys()) == expected_keys


def test_compute_all_metrics_perfect_classifier():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.98, 0.99])
    result = compute_all_metrics(y_true, y_prob)
    assert result["auc_roc"] == 1.0
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["brier_score"] < 0.01


def test_compute_all_metrics_custom_threshold(binary_labels):
    y_true, y_prob = binary_labels
    result = compute_all_metrics(y_true, y_prob, threshold=0.5)
    assert result["threshold_used"] == 0.5


def test_compute_all_metrics_confusion_matrix(binary_labels):
    y_true, y_prob = binary_labels
    result = compute_all_metrics(y_true, y_prob)
    cm = result["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2
    # Rows = actual, cols = predicted; total must equal n
    assert cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] == len(y_true)


def test_find_optimal_threshold_youden(binary_labels):
    y_true, y_prob = binary_labels
    threshold = find_optimal_threshold(y_true, y_prob, method="youden")
    assert 0.0 < threshold < 1.0


def test_find_f1_threshold(binary_labels):
    y_true, y_prob = binary_labels
    threshold = find_f1_threshold(y_true, y_prob)
    assert 0.0 < threshold < 1.0


def test_temporal_iou_full_overlap():
    t = pd.Timestamp
    iou = temporal_iou(t("2024-01-01"), t("2024-01-31"),
                       t("2024-01-01"), t("2024-01-31"))
    assert iou == pytest.approx(1.0)


def test_temporal_iou_no_overlap():
    t = pd.Timestamp
    iou = temporal_iou(t("2024-01-01"), t("2024-01-15"),
                       t("2024-01-16"), t("2024-01-31"))
    assert iou == pytest.approx(0.0)


def test_temporal_iou_partial():
    t = pd.Timestamp
    iou = temporal_iou(t("2024-01-01"), t("2024-01-20"),
                       t("2024-01-10"), t("2024-01-31"))
    assert 0.0 < iou < 1.0


# ── registry helpers ──────────────────────────────────────────────────────────

def test_registry_total_count():
    assert len(FEATURE_REGISTRY) == 57


def test_get_all_feature_names_count():
    names = get_all_feature_names()
    assert len(names) == 57


def test_get_features_by_group_velocity():
    feats = get_features_by_group("velocity")
    assert len(feats) == 10
    assert "txn_count_7d" in feats


def test_get_features_by_group_all_groups():
    groups = ["velocity", "amount_patterns", "temporal", "passthrough",
              "graph_network", "profile_mismatch", "kyc_behavioral", "interactions"]
    expected = [10, 8, 8, 7, 10, 5, 4, 5]
    for grp, exp in zip(groups, expected):
        assert len(get_features_by_group(grp)) == exp, f"{grp} expected {exp}"


def test_get_high_power_features_nonempty():
    high = get_high_power_features()
    assert len(high) > 0
    assert all(FEATURE_REGISTRY[f]["power"] == "High" for f in high)


def test_print_feature_summary_runs(capsys):
    print_feature_summary()
    captured = capsys.readouterr()
    assert "57" in captured.out
    assert "velocity" in captured.out


def test_registry_each_entry_has_required_keys():
    required = {"group", "description", "dtype", "power"}
    for name, meta in FEATURE_REGISTRY.items():
        assert required.issubset(meta.keys()), f"{name} missing keys"
        assert meta["power"] in ("High", "Medium"), f"{name} bad power"


# ── logging_config ────────────────────────────────────────────────────────────

def test_setup_logger_returns_logger():
    import logging
    logger = setup_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_module"


def test_setup_logger_idempotent():
    l1 = setup_logger("idempotent_test")
    handler_count = len(l1.handlers)
    l2 = setup_logger("idempotent_test")  # second call
    assert len(l2.handlers) == handler_count  # no duplicate handlers


# ── preprocess_profile ────────────────────────────────────────────────────────

def test_preprocess_profile_fills_numeric_nulls():
    import pandas as pd
    profile = pd.DataFrame({
        "account_id": ["A", "B", "C"],
        "declared_income": [100000.0, None, 200000.0],
        "account_age_days": [365.0, 730.0, None],
    })
    result = preprocess_profile(profile)
    assert result["declared_income"].isnull().sum() == 0
    assert result["account_age_days"].isnull().sum() == 0
    # Null filled with median: median([100000, 200000]) = 150000
    assert result.loc[1, "declared_income"] == pytest.approx(150000.0)


def test_preprocess_profile_fills_categorical_nulls():
    import pandas as pd
    profile = pd.DataFrame({
        "account_id": ["A", "B"],
        "account_type": ["savings", None],
    })
    result = preprocess_profile(profile)
    assert result["account_type"].isnull().sum() == 0
    assert result.loc[1, "account_type"] == "UNKNOWN"
