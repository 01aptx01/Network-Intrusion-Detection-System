import numpy as np
import logging

class NIDSEvaluator:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        self.y_true = y_true.flatten()
        self.y_pred = y_pred.flatten()
        self._compute_confusion_matrix()

    def _compute_confusion_matrix(self):
        self.TP = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.TN = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.FP = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.FN = np.sum((self.y_true == 1) & (self.y_pred == 0))

    def report(self, logger: logging.Logger = None):
        p = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0
        r = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        
        report_str = (
            f"\n{'='*40}\n"
            f"📊 NIDS EVALUATION REPORT\n"
            f"{'='*40}\n"
            f"TP : {self.TP} (Caught Attacks)\n"
            f"TN : {self.TN} (Normal Traffic)\n"
            f"FP : {self.FP} (False Alarms)\n"
            f"FN : {self.FN} <--- 🚨 Missed Attacks!\n"
            f"{'-'*40}\n"
            f"Precision : {p:.4f}\n"
            f"Recall    : {r:.4f}\n"
            f"F1-Score  : {f1:.4f}\n"
            f"{'='*40}"
        )
        
        if logger:
            logger.info(report_str)
        else:
            print(report_str)