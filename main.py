import numpy as np
import config
import pandas as pd
from src.utils import NIDSUtils
from src.preprocessing import NIDSDataPreprocessor
from src.model import CustomLogisticRegression
from src.evaluator import NIDSEvaluator

def generate_mock_data():
    """ใช้ในกรณีที่คุณยังโหลดไฟล์ NSL-KDD CSV มาใส่โฟลเดอร์ data ไม่เป็น"""
    np.random.seed(42)
    X_train = np.random.randn(5000, 41) * 50
    y_train = np.random.choice([0, 1], size=5000, p=[0.8, 0.2])
    X_test = np.random.randn(1000, 41) * 50
    y_test = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    return X_train, y_train, X_test, y_test

def analyze_feature_importance(W: np.ndarray, feature_names: pd.Index, logger=None):
    """
    วิเคราะห์และตีความพฤติกรรมของโมเดล (Model Interpretability)
    """
    # 1. ทำให้เป็นเวกเตอร์ 1 มิติ (M,) เพื่อให้สอดคล้องกับ Array ของชื่อคอลัมน์
    W_flat = W.flatten()
    
    # 2. หา Index ที่เรียงลำดับจากค่าน้อยสุด (ลบมากสุด) ไปหาค่ามากสุด (บวกมากสุด)
    # Time Complexity: O(M log M)
    sorted_indices = np.argsort(W_flat)
    
    report = ["\n" + "="*60]
    report.append("🔍 FEATURE IMPORTANCE ANALYSIS (INTERPRETABILITY)")
    report.append("="*60)
    
    # Top 5 ฟีเจอร์ที่บ่งบอกถึงการ "โจมตี" (ค่า Weight เป็นบวกสูงที่สุด)
    # ตัดเอา 5 ตัวสุดท้าย แล้วใช้ [::-1] เพื่อกลับด้านให้ตัวที่บวกมากที่สุดขึ้นก่อน
    top_attack_idx = sorted_indices[-5:][::-1]
    
    report.append("🚨 TOP 5 ATTACK INDICATORS (High Positive Weights):")
    report.append("   (ยิ่งฟีเจอร์เหล่านี้มีค่าสูง โมเดลยิ่งฟันธงว่าถูกแฮ็ก)")
    for i, idx in enumerate(top_attack_idx):
        report.append(f"   {i+1}. {str(feature_names[idx]):<30} : {W_flat[idx]:+.4f}")
        
    report.append("-" * 60)
    
    # Top 5 ฟีเจอร์ที่บ่งบอกถึงความ "ปลอดภัย" (ค่า Weight เป็นลบต่ำที่สุด)
    # ตัดเอา 5 ตัวแรกสุดจาก Array ที่เรียงแล้ว
    top_normal_idx = sorted_indices[:5]
    
    report.append("🛡️ TOP 5 NORMAL INDICATORS (High Negative Weights):")
    report.append("   (ยิ่งฟีเจอร์เหล่านี้มีค่าสูง โมเดลยิ่งมั่นใจว่าเป็น Traffic ปกติ)")
    for i, idx in enumerate(top_normal_idx):
        report.append(f"   {i+1}. {str(feature_names[idx]):<30} : {W_flat[idx]:+.4f}")
        
    report.append("="*60)
    
    # พิมพ์ออกจอหรือบันทึกลง Log
    output = "\n".join(report)
    if logger:
        logger.info(output)
    else:
        print(output)

def main():
    logger = NIDSUtils.setup_logger(config.LOG_DIR)
    logger.info("🚀 Starting NIDS Pipeline...")

    preprocessor = NIDSDataPreprocessor()

    try:
        # พยายามโหลดไฟล์จริงก่อน
        logger.info(f"Loading training data from {config.DATA_PATH_TRAIN}...")
        X_train, y_train = preprocessor.load_data(config.DATA_PATH_TRAIN, is_train=True)
        X_test, y_test = preprocessor.load_data(config.DATA_PATH_TEST, is_train=False)
    except FileNotFoundError:
        logger.warning("⚠️ Real CSV data not found! Falling back to Mock Data generation.")
        X_train, y_train, X_test, y_test = generate_mock_data()

    logger.info("Applying Z-Score Normalization...")
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    weights = preprocessor.calculate_class_weights(y_train)
    attack_penalty = weights[1] / weights[0]
    logger.info(f"Class Imbalance Penalty Weight (Attack vs Normal): {attack_penalty:.2f}x")

    logger.info("Initializing & Training Model...")
    model = CustomLogisticRegression(
        learning_rate=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE
    )
    
    # ส่ง logger เข้าไปเพื่อให้ปรินต์ Loss ในแต่ละ Epoch ลงไฟล์ได้
    model.fit(X_train_scaled, y_train, weight_attack=attack_penalty, logger=logger)

    logger.info("Saving trained weights to disk...")
    NIDSUtils.save_model(model.W, model.b, config.MODEL_SAVE_PATH)

    logger.info("Saving preprocessor state to disk...")
    NIDSUtils.save_preprocessor(preprocessor, config.MODEL_SAVE_PATH.replace('.npz', '_preprocessor.pkl'))

    logger.info(f"Evaluating with Decision Threshold = {config.DECISION_THRESHOLD}...")
    y_pred = model.predict(X_test_scaled, threshold=config.DECISION_THRESHOLD)
    
    evaluator = NIDSEvaluator(y_test, y_pred)
    evaluator.report(logger=logger)

    logger.info("Extracting Feature Importance...")
    analyze_feature_importance(model.W, preprocessor.train_columns, logger=logger)
    
    logger.info("✅ Pipeline Execution Finished.")

if __name__ == "__main__":
    import sys

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, AttributeError):
                pass
    main()