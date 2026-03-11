import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, brier_score_loss, confusion_matrix,
)


def temporal_iou(pred_start, pred_end, true_start, true_end) -> float:
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    intersection = max(0, (intersection_end - intersection_start).total_seconds())
    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union = max(0, (union_end - union_start).total_seconds())
    return intersection / union if union > 0 else 0.0


def find_optimal_threshold(y_true, y_prob, method: str = "youden") -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if method == "youden":
        idx = np.argmax(tpr - fpr)
    else:
        idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


def find_f1_threshold(y_true, y_prob) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    idx = np.argmax(f1_scores[:-1])
    return float(thresholds[idx])


def compute_all_metrics(y_true, y_prob, threshold: float = None) -> dict:
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)

    y_pred = (np.array(y_prob) >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "brier_score": brier,
        "confusion_matrix": cm,
        "threshold_used": threshold,
    }
