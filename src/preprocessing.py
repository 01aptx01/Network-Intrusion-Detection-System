import numpy as np
import pandas as pd

class NIDSDataPreprocessor:
    def __init__(self):
        """
        เก็บ State ของโมเดลไว้เพื่อไม่ให้เกิด Data Leakage
        เราต้องใช้ Mean และ Std จาก Train set ไปประยุกต์ใช้กับ Test set
        """
        self.mean = None
        self.std = None
        self.is_fitted = False

    @staticmethod
    def load_data(filepath: str, is_train: bool = True) -> tuple:
        """
        โหลดข้อมูล NSL-KDD และแยก Features (X) กับ Labels (y)
        Time Complexity: O(N * M) - N คือจำนวนแถว, M คือจำนวนคอลัมน์ (การอ่านไฟล์ลง Memory)
        Space Complexity: O(N * M) - ขนาดของ DataFrame และ NumPy Arrays
        """
        # อนุญาตให้ใช้ Pandas เพื่อความสะดวกในการจัดการ CSV
        df = pd.read_csv(filepath)
        
        # สมมติว่าคอลัมน์เป้าหมายชื่อ 'label' หรือ 'class'
        # NSL-KDD ต้นฉบับมักจะมีค่าเป็นข้อความ เช่น 'normal', 'neptune', 'smurf' ฯลฯ
        # เราจะต้องแปลง 'normal' เป็น 0 และการโจมตีทั้งหมดเป็น 1
        target_col = 'label' if 'label' in df.columns else df.columns[-1]
        
        # 0 = Normal, 1 = Attack (Binary Classification)
        y = np.where(df[target_col] == 'normal', 0, 1)
        
        # ตัดคอลัมน์ Label ทิ้งเพื่อเอาเฉพาะ Features
        X_df = df.drop(columns=[target_col])
        
        # แปลงเป็น Matrix บริสุทธิ์ (ลบ Header ทิ้ง)
        X = X_df.values.astype(np.float64)
        
        return X, y

    def fit(self, X: np.ndarray) -> None:
        """
        เรียนรู้การกระจายตัวของข้อมูล (คำนวณ μ และ σ)
        Time Complexity: O(N * M) 
        Space Complexity: O(M) - เก็บค่า Array ขนาดเท่ากับจำนวน Features
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # ป้องกันหายนะทางคณิตศาสตร์: Division by Zero
        # หาก Feature ใดมีค่าคงที่ (เช่น เป็น 0 ทุกแถว) std จะเป็น 0
        # เราต้องบวกค่า Epsilon เล็กๆ เข้าไปเพื่อไม่ให้ระบบพังตอนหาร
        self.std[self.std == 0] = 1e-8
        
        self.is_fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        แปลงข้อมูลให้อยู่ในสเกล Z-score
        Time Complexity: O(N * M) - Matrix Element-wise Operation
        Space Complexity: O(N * M) - สร้าง Matrix ผลลัพธ์ก้อนใหม่
        """
        if not self.fitted:
            raise RuntimeError("คุณต้องเรียก .fit() ก่อนที่จะ .transform() เสมอ!")
        
        # Z = (X - μ) / σ
        X_scaled = (X - self.mean) / self.std
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience method สำหรับ Train set"""
        self.fit(X)
        return self.transform(X)

    @staticmethod
    def calculate_class_weights(y: np.ndarray) -> dict:
        """
        คำนวณ Class Weights ตามสัดส่วนความไม่สมดุลของข้อมูล
        Time Complexity: O(N) - สแกนหาความถี่ของแต่ละคลาส
        Space Complexity: O(K) - เก็บ Dict ตามจำนวนคลาส (K=2)
        """
        n_samples = len(y)
        classes, counts = np.unique(y, return_counts=True)
        
        weights = {}
        for cls, count in zip(classes, counts):
            # สมการ: W_j = N / (K * n_j)
            weights[cls] = n_samples / (len(classes) * count)
            
        return weights