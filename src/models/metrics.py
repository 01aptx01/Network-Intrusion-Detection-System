"""Model evaluation metrics calculation and reporting."""

import logging
from typing import Optional

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class BinaryClassifierMetrics:
    """คำนวณเมทริกสำหรับ binary classification ด้วย scikit-learn"""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true.flatten()
        self.y_pred = y_pred.flatten()

    def log_report(self, logger: Optional[logging.Logger] = None) -> None:
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        precision = precision_score(self.y_true, self.y_pred, zero_division=0)
        recall = recall_score(self.y_true, self.y_pred, zero_division=0)
        f1 = f1_score(self.y_true, self.y_pred, zero_division=0)

        report_str = (
            f"\n{'='*40}\n"
            f"Binary classification report (scikit-learn)\n"
            f"{'='*40}\n"
            f"TP : {tp} (caught attacks)\n"
            f"TN : {tn} (normal traffic)\n"
            f"FP : {fp} (false alarms)\n"
            f"FN : {fn} (missed attacks)\n"
            f"{'-'*40}\n"
            f"Precision : {precision:.4f}\n"
            f"Recall    : {recall:.4f}\n"
            f"F1-Score  : {f1:.4f}\n"
            f"{'='*40}"
        )

        if logger is not None:
            logger.info(report_str)
        else:
            print(report_str)
