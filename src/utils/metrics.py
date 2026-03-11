import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def temporal_iou(pred_start, pred_end, true_start, true_end) -> float:
    overlap_start = max(pred_start, true_start)
    overlap_end = min(pred_end, true_end)
    intersection = max(0, (overlap_end - overlap_start).total_seconds()
                       if hasattr(overlap_end, "total_seconds")
                       else float(overlap_end - overlap_start))
    union_start = min(pred_start, true_start)
    union_end = max(pred_end, true_end)
    union = max(1e-9, (union_end - union_start).total_seconds()
                if hasattr(union_end, "total_seconds")
                else float(union_end - union_start))
    return intersection / union


def find_optimal_threshold(y_true, y_prob, method="youden") -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    if method == "youden":
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    raise ValueError(f"Unknown method: {method}")


def find_f1_threshold(y_true, y_prob) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx])


def compute_all_metrics(y_true, y_prob, threshold=None) -> dict:
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_prob)
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "threshold_used": threshold,
    }
