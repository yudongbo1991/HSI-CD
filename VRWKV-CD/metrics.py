"""Binary change-detection metrics; internal class 0 is changed."""
import numpy as np
from sklearn.metrics import confusion_matrix


def binary_metrics(target, prediction):
    matrix = confusion_matrix(target, prediction, labels=[0, 1]).astype(np.int64)
    tp, fn = matrix[0]
    fp, tn = matrix[1]
    total = matrix.sum()
    oa = (tp + tn) / total
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / total**2
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return {
        "oa": float(oa), "kappa": float((oa - expected) / max(1 - expected, 1e-12)),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "iou": float(iou), "total": int(total),
        "confusion": [[int(tp), int(fn)], [int(fp), int(tn)]],
    }
