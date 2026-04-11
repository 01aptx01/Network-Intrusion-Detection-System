import numpy as np

class NIDSEvaluator:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        รับค่า True Labels และ Predicted Labels แบบ 1D Array (N,)
        """
        self.y_true = y_true.flatten()
        self.y_pred = y_pred.flatten()
        
        # ป้องกันมิติข้อมูลไม่ตรงกัน
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("มิติของ y_true และ y_pred ต้องเท่ากัน!")
            
        self._compute_confusion_matrix()

    def _compute_confusion_matrix(self) -> None:
        """
        คำนวณ TP, TN, FP, FN ด้วย Vectorized Bitwise Operations
        Time Complexity: O(N) - เทียบค่า Boolean ใน Array ขนาด N แบบขนาน
        Space Complexity: O(1) - เก็บแค่ตัวเลข Integer 4 ตัว
        """
        # ใช้ Bitwise AND (&) เพื่อหาเงื่อนไขที่ตรงกันทั้งคู่
        self.TP = np.sum((self.y_true == 1) & (self.y_pred == 1))
        self.TN = np.sum((self.y_true == 0) & (self.y_pred == 0))
        self.FP = np.sum((self.y_true == 0) & (self.y_pred == 1))
        self.FN = np.sum((self.y_true == 1) & (self.y_pred == 0))

    def precision(self) -> float:
        if self.TP + self.FP == 0: return 0.0
        return self.TP / (self.TP + self.FP)

    def recall(self) -> float:
        if self.TP + self.FN == 0: return 0.0
        return self.TP / (self.TP + self.FN)

    def f1_score(self) -> float:
        p = self.precision()
        r = self.recall()
        if p + r == 0: return 0.0
        return 2 * (p * r) / (p + r)

    def report(self) -> None:
        print("\n" + "="*30)
        print("📊 NIDS Evaluation Report")
        print("="*30)
        print(f"True Positives (TP):  {self.TP}")
        print(f"True Negatives (TN):  {self.TN}")
        print(f"False Positives (FP): {self.FP} (False Alarms)")
        print(f"False Negatives (FN): {self.FN} (Missed Attacks) <- 🚨 FOCUS HERE")
        print("-" * 30)
        print(f"Precision: {self.precision():.4f}")
        print(f"Recall:    {self.recall():.4f}")
        print(f"F1-Score:  {self.f1_score():.4f}")
        print("="*30)