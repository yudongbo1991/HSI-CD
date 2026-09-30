from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def binary_metrics(target, prediction):
    matrix = confusion_matrix(target, prediction, labels=[0, 1]).astype(np.int64)
    total = matrix.sum()
    oa = float(np.trace(matrix) / total)
    recall = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
    aa = float(recall.mean())
    expected = float((matrix.sum(0) * matrix.sum(1)).sum() / (total * total))
    kappa = float((oa - expected) / max(1.0 - expected, 1e-12))
    intersection = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    iou = intersection / np.maximum(union, 1)
    precision = np.diag(matrix) / np.maximum(matrix.sum(0), 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "oa": oa, "aa": aa, "kappa": kappa,
        "miou": float(iou.mean()), "change_iou": float(iou[0]),
        "macro_f1": float(f1.mean()), "confusion": matrix.tolist(),
        "correct": int(np.trace(matrix)), "total": int(total),
        "change_ok": int(matrix[0, 0]), "miss": int(matrix[0, 1]),
        "false_alarm": int(matrix[1, 0]), "unchanged_ok": int(matrix[1, 1]),
    }


def format_metrics(prefix: str, epoch: int, values: dict) -> str:
    return (f"{prefix} Epoch: {epoch:03d} | OA: {values['oa']:.8f} | "
            f"AA: {values['aa']:.6f} | Kappa: {values['kappa']:.6f} | "
            f"mIoU: {values['miou']:.6f} | IoU(change): {values['change_iou']:.6f} | "
            f"F1: {values['macro_f1']:.6f} | correct: "
            f"{values['correct']}/{values['total']} | CHG_OK: {values['change_ok']} "
            f"MISS: {values['miss']} FA: {values['false_alarm']} "
            f"UNCH_OK: {values['unchanged_ok']}")
