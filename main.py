import sys

import numpy as np
import pandas as pd

import config
from src.artifacts import ArtifactStore
from src.metrics import BinaryClassifierMetrics
from src.model import BinaryLogisticRegression
from src.preprocessing import IntrusionDatasetPreprocessor


def synthesize_fallback_dataset():
    """สร้างข้อมูลสุ่มเมื่อยังไม่มีไฟล์ NSL-KDD ใน data/raw"""
    np.random.seed(42)
    features_train = np.random.randn(5000, 41) * 50
    labels_train = np.random.choice([0, 1], size=5000, p=[0.8, 0.2])
    features_test = np.random.randn(1000, 41) * 50
    labels_test = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    return features_train, labels_train, features_test, labels_test


def analyze_feature_importance(weights: np.ndarray, feature_names: pd.Index, logger=None):
    """เรียง weight เพื่อดูทิศทางฟีเจอร์ที่ผลักดันไปคลาส attack vs normal"""
    weights_flat = weights.flatten()
    sorted_indices = np.argsort(weights_flat)

    lines = ["\n" + "=" * 60, "Feature importance (linear weights)", "=" * 60]
    top_attack_idx = sorted_indices[-5:][::-1]
    lines.append("Top 5 attack indicators (most positive weights):")
    for i, idx in enumerate(top_attack_idx):
        lines.append(f"   {i + 1}. {str(feature_names[idx]):<30} : {weights_flat[idx]:+.4f}")

    lines.append("-" * 60)
    top_normal_idx = sorted_indices[:5]
    lines.append("Top 5 normal indicators (most negative weights):")
    for i, idx in enumerate(top_normal_idx):
        lines.append(f"   {i + 1}. {str(feature_names[idx]):<30} : {weights_flat[idx]:+.4f}")

    lines.append("=" * 60)
    output = "\n".join(lines)
    if logger:
        logger.info(output)
    else:
        print(output)


def main():
    logger = ArtifactStore.configure_logger(config.LOG_DIR)
    logger.info("Starting intrusion-detection training pipeline...")

    preprocessor = IntrusionDatasetPreprocessor()

    try:
        logger.info(f"Loading training data from {config.DATA_PATH_TRAIN}...")
        features_train, labels_train = preprocessor.load_data(config.DATA_PATH_TRAIN, is_train=True)
        features_test, labels_test = preprocessor.load_data(config.DATA_PATH_TEST, is_train=False)
    except FileNotFoundError:
        logger.warning("Training data not found; using synthesized fallback dataset.")
        features_train, labels_train, features_test, labels_test = synthesize_fallback_dataset()

    logger.info("Applying Z-score normalization...")
    features_train_scaled = preprocessor.fit_transform(features_train)
    features_test_scaled = preprocessor.transform(features_test)

    class_weights = preprocessor.compute_inverse_frequency_class_weights(labels_train)
    attack_loss_weight = class_weights[1] / class_weights[0]
    logger.info(f"Class imbalance factor (attack vs normal): {attack_loss_weight:.2f}x")

    logger.info("Training binary logistic regression...")
    model = BinaryLogisticRegression(
        learning_rate=config.LEARNING_RATE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
    )

    model.fit(
        features_train_scaled,
        labels_train,
        attack_loss_weight=attack_loss_weight,
        logger=logger,
    )

    logger.info("Saving model weights...")
    ArtifactStore.save_model_weights(model.weights, model.bias, config.MODEL_SAVE_PATH)

    logger.info("Saving preprocessor state...")
    preprocessor_path = config.MODEL_SAVE_PATH.replace(".npz", "_preprocessor.pkl")
    ArtifactStore.save_preprocessor(preprocessor, preprocessor_path)

    logger.info(f"Evaluating at decision threshold = {config.DECISION_THRESHOLD}...")
    predicted_labels = model.predict(features_test_scaled, decision_threshold=config.DECISION_THRESHOLD)

    metrics = BinaryClassifierMetrics(labels_test, predicted_labels)
    metrics.log_report(logger=logger)

    logger.info("Feature importance summary...")
    analyze_feature_importance(model.weights, preprocessor.train_column_names, logger=logger)

    logger.info("Pipeline finished.")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, AttributeError):
                pass
    main()
