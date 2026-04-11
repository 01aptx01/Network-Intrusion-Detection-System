import numpy as np
import pandas as pd

class NIDSDataPreprocessor:
    def __init__(self):
        self.mean = None
        self.std = None
        self.is_fitted = False
        self.train_columns = None # จำโครงสร้างคอลัมน์ตอนทำ One-Hot

    def load_data(self, filepath: str, is_train: bool = True) -> tuple:
        """
        อ่าน CSV และทำ One-Hot Encoding สำหรับ Categorical Features
        อนุญาตให้ใช้ Pandas เฉพาะขั้นตอนนี้
        """
        try:
            df = pd.read_csv(filepath, header=None)
        except FileNotFoundError:
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลที่: {filepath}")

        # สมมติว่าคอลัมน์สุดท้ายคือ Label (ปรับชื่อตามจริง)
        if df.shape[1] >= 43:
            df = df.drop(columns=[42])

        target_col = 41
        
        # 0 = normal, 1 = attack
        y = np.where(df[target_col] == 'normal', 0, 1)
        X_df = df.drop(columns=[target_col])

        # แปลงข้อมูลข้อความ (เช่น tcp, udp) เป็น One-Hot Vectors [0, 1, 0]
        # เพื่อป้องกันการเกิด False Ordinality ทางคณิตศาสตร์
        X_df = pd.get_dummies(X_df)

        if is_train:
            # จำรายชื่อคอลัมน์ที่เกิดจาก Train set ไว้
            self.train_columns = X_df.columns
        else:
            # สำหรับ Test set ต้องบังคับให้คอลัมน์ตรงกับ Train set เสมอ
            # ถ้า Test set มีชนิดข้อมูลแปลกๆ โผล่มา หรือหายไป จะได้ไม่พังตอนคูณเมทริกซ์
            X_df = X_df.reindex(columns=self.train_columns, fill_value=0)

        return X_df.values.astype(np.float64), y

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Z-score Standardization 
        Time Complexity: O(N * M)
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-8 # กันสมการระเบิด
        self.is_fitted = True
        return (X - self.mean) / self.std

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ต้อง fit() ก่อนเสมอ!")
        return (X - self.mean) / self.std

    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> dict:
        """คำนวณ Penalty Weights แบบ Vectorized"""
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        return {cls: n_samples / (len(classes) * count) for cls, count in zip(classes, counts)}