"""Script to train specified models on dataset."""

import sys
import numpy as np
from sklearn.pipeline import Pipeline

from src import config
from src.data.preprocessing import IntrusionDatasetPreprocessor, synthesize_fallback_dataset
from src.features.build_features import build_feature_pipeline
from src.models.artifacts import ArtifactStore
from src.models.metrics import BinaryClassifierMetrics
from src.models.model import ModelFactory
from src.visualization.visualize import analyze_feature_importance


def train_pipeline():
    """Main pipeline for training and evaluating the model.
    
    This function handles the end-to-end process:
    1. Loading the dataset (or synthetic fallback).
    2. Building the preprocessing pipeline.
    3. Instantiating the model according to the config.
    4. Training the Scikit-learn Pipeline (preprocessing + classifier).
    5. Saving the trained model.
    6. Evaluating the model and logging metrics (Precision, Recall, etc.).
    
    Returns:
        Pipeline: The trained scikit-learn Pipeline object.
    """
    logger = ArtifactStore.configure_logger(config.LOG_DIR)
    logger.info("Starting intrusion-detection training pipeline (scikit-learn)...")

    preprocessor = IntrusionDatasetPreprocessor()

    try:
        logger.info(f"Loading training data from {config.DATA_PATH_TRAIN}...")
        features_train, labels_train = preprocessor.load_data(config.DATA_PATH_TRAIN)
        features_test, labels_test = preprocessor.load_data(config.DATA_PATH_TEST)
    except FileNotFoundError:
        logger.warning("Training data not found; using synthesized fallback dataset.")
        features_train, labels_train, features_test, labels_test = (
            synthesize_fallback_dataset()
        )

    logger.info("Creating preprocessor transformer...")
    transformer = build_feature_pipeline(features_train)

    logger.info(f"Instantiating model: {config.MODEL_CONFIG['type']}...")
    model = ModelFactory.get_model(config.MODEL_CONFIG)

    pipeline = Pipeline(steps=[("preprocessor", transformer), ("classifier", model)])

    logger.info("Training pipeline...")
    pipeline.fit(features_train, labels_train)

    logger.info(f"Saving pipeline to {config.MODEL_SAVE_PATH}...")
    ArtifactStore.save_pipeline(pipeline, config.MODEL_SAVE_PATH)

    logger.info("Evaluating...")
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(features_test)[:, 1]
        predicted_labels = (probas >= config.DECISION_THRESHOLD).astype(int)
    else:
        predicted_labels = pipeline.predict(features_test)

    metrics = BinaryClassifierMetrics(labels_test, predicted_labels)
    metrics.log_report(logger=logger)

    analyze_feature_importance(
        pipeline.named_steps["classifier"],
        pipeline.named_steps["preprocessor"],
        logger=logger,
    )

    logger.info("Pipeline finished.")
    return pipeline


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, AttributeError):
                pass
    train_pipeline()
