import numpy as np
import config
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

    logger.info(f"Evaluating with Decision Threshold = {config.DECISION_THRESHOLD}...")
    y_pred = model.predict(X_test_scaled, threshold=config.DECISION_THRESHOLD)
    
    evaluator = NIDSEvaluator(y_test, y_pred)
    evaluator.report(logger=logger)
    
    logger.info("✅ Pipeline Execution Finished.")

if __name__ == "__main__":
    main()