"""เมทริกสองคลาสจาก confusion matrix (ไม่ใช้ scikit-learn)"""
import logging
from typing import Optional

import numpy as np


class BinaryClassifierMetrics:
    """คำนวณ TP/TN/FP/FN และ precision, recall, F1 สำหรับคลาสบวก (attack = 1)"""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true.flatten()
        self.y_pred = y_pred.flatten()
        self._compute_confusion_matrix()

    def _compute_confusion_matrix(self) -> None:
        self.true_positives = int(np.sum((self.y_true == 1) & (self.y_pred == 1)))
        self.true_negatives = int(np.sum((self.y_true == 0) & (self.y_pred == 0)))
        self.false_positives = int(np.sum((self.y_true == 0) & (self.y_pred == 1)))
        self.false_negatives = int(np.sum((self.y_true == 1) & (self.y_pred == 0)))

    def log_report(self, logger: Optional[logging.Logger] = None) -> None:
        tp, tn, fp, fn = self.true_positives, self.true_negatives, self.false_positives, self.false_negatives
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        report_str = (
            f"\n{'='*40}\n"
            f"Binary classification report\n"
            f"{'='*40}\n"
            f"TP : {tp} (caught attacks)\n"
            f"TN : {tn} (normal traffic)\n"
            f"FP : {fp} (false alarms)\n"
            f"FN : {fn} (missed attacks)\n"
            f"{'-'*40}\n"
            f"Precision : {precision:.4f}\n"
            f"Recall    : {recall:.4f}\n"
            f"F1-Score  : {f1_score:.4f}\n"
            f"{'='*40}"
        )

        if logger is not None:
            logger.info(report_str)
        else:
            print(report_str)
