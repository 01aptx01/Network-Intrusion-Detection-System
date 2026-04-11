import numpy as np
import pandas as pd
from src.preprocessing import NIDSDataPreprocessor
from src.model import CustomLogisticRegression
from src.evaluator import NIDSEvaluator

def load_data(filepath: str):
    """
    อนุญาตให้ใช้ Pandas เฉพาะตอนอ่าน CSV ขึ้นมาเป็น DataFrame 
    จากนั้นต้องแปลงเป็น NumPy Array ทันที
    """
    print(f"[*] Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    
    # สมมติว่าคอลัมน์สุดท้ายชื่อ 'label' (0=Normal, 1=Attack)
    X = df.drop('label', axis=1).values 
    y = df['label'].values
    return X, y

def main():
    # 1. โหลดข้อมูล (ในสถานการณ์จริง เปลี่ยนเป็น Path ของ NSL-KDD)
    # X_train, y_train = load_data("data/raw/KDDTrain.csv")
    # X_test, y_test = load_data("data/raw/KDDTest.csv")
    
    # [Mock Data สำหรับการทดสอบโค้ด: 1000 samples, 41 features]
    print("[*] Generating Mock Data for Pipeline Testing...")
    np.random.seed(42)
    X_train = np.random.randn(1000, 41) * 100 
    y_train = np.random.choice([0, 1], size=1000, p=[0.8, 0.2]) # Imbalance 80:20
    X_test = np.random.randn(200, 41) * 100
    y_test = np.random.choice([0, 1], size=200, p=[0.8, 0.2])

    # 2. Data Preprocessing (ต้อง fit แค่ Train set เพื่อกัน Data Leakage!)
    print("[*] Initializing Preprocessor and scaling data...")
    preprocessor = NIDSDataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    # คำนวณ Class Weights ตามหลักคณิตศาสตร์ที่เราคุยกัน
    weights = preprocessor.calculate_class_weights(y_train)
    weight_attack = weights[1] / weights[0] # หา Ratio ของน้ำหนัก
    print(f"[*] Calculated Attack Penalty Weight: {weight_attack:.2f}x")

    # 3. Model Initialization & Training
    print("[*] Initializing Custom Logistic Regression Model...")
    model = CustomLogisticRegression(learning_rate=0.01, epochs=2000)
    
    print("[*] Training Model with Gradient Descent (This might take a moment)...")
    # หมายเหตุ: ใน model.py คุณต้องนำ weight_attack ไปใช้ตามที่ผมสอนไปก่อนหน้า
    model.fit(X_train_scaled, y_train, weight_attack=weight_attack)

    # 4. Evaluation
    print("[*] Predicting on Test Set...")
    # ทายผลลัพธ์ (ได้เป็น 0 หรือ 1)
    y_pred = model.predict(X_test_scaled, threshold=0.5)

    print("[*] Running Evaluation Metrics...")
    evaluator = NIDSEvaluator(y_test, y_pred)
    evaluator.report()

if __name__ == "__main__":
    main()