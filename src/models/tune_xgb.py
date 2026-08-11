"""Script for tuning XGBoost hyperparameters."""

import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from src import config
from src.data.preprocessing import IntrusionDatasetPreprocessor
from src.features.build_features import build_feature_pipeline
from src.models.artifacts import ArtifactStore

def tune_xgboost():
    logger = ArtifactStore.configure_logger(config.LOG_DIR)
    logger.info("Starting XGBoost hyperparameter tuning...")

    preprocessor = IntrusionDatasetPreprocessor()
    logger.info(f"Loading training data...")
    features_train, labels_train = preprocessor.load_data(config.DATA_PATH_TRAIN)
    features_test, labels_test = preprocessor.load_data(config.DATA_PATH_TEST)

    transformer = build_feature_pipeline(features_train)

    # 10 predefined configurations for XGBoost
    param_grid = [
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "scale_pos_weight": 1}, # Default
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "scale_pos_weight": 5}, # Higher weight for minority
        {"n_estimators": 100, "max_depth": 10, "learning_rate": 0.1, "scale_pos_weight": 10}, # Deeper, higher weight
        {"n_estimators": 200, "max_depth": 10, "learning_rate": 0.05, "scale_pos_weight": 5}, # Slower learning
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.01, "scale_pos_weight": 10}, # Very slow learning
        {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.2, "scale_pos_weight": 1}, # Shallow, fast learning
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "scale_pos_weight": 20}, # Very high weight
        {"n_estimators": 150, "max_depth": 8, "learning_rate": 0.05, "scale_pos_weight": 10}, # Balanced
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3, "scale_pos_weight": 5}, # Faster learning
        {"n_estimators": 250, "max_depth": 12, "learning_rate": 0.05, "scale_pos_weight": 10}, # Complex model
    ]

    best_score = 0
    best_params = None
    best_metrics = {}
    
    results = []

    for i, params in enumerate(param_grid):
        logger.info(f"--- Testing config {i+1}/10: {params} ---")
        
        # Build pipeline
        model = XGBClassifier(random_state=42, n_jobs=-1, **params)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("feature_selection", SelectPercentile(score_func=f_classif, percentile=50)),
                ("classifier", model),
            ]
        )
        
        # Train
        pipeline.fit(features_train, labels_train)
        
        # Predict on Test Set
        if hasattr(pipeline, "predict_proba"):
            probas = pipeline.predict_proba(features_test)[:, 1]
            predicted_labels = (probas >= config.DECISION_THRESHOLD).astype(int)
        else:
            predicted_labels = pipeline.predict(features_test)
        
        # Evaluate
        f1_binary = f1_score(labels_test, predicted_labels, average="binary")
        f1_macro = f1_score(labels_test, predicted_labels, average="macro")
        precision = precision_score(labels_test, predicted_labels, average="macro")
        recall = recall_score(labels_test, predicted_labels, average="macro")
        
        logger.info(f"Config {i+1} Results -> F1-Macro: {f1_macro:.4f} | Recall (Macro): {recall:.4f}")
        
        results.append({
            "config_id": i+1,
            "params": params,
            "f1_binary": f1_binary,
            "f1_macro": f1_macro,
            "precision_macro": precision,
            "recall_macro": recall
        })
        
        # Select best based on F1-Macro
        if f1_macro > best_score:
            best_score = f1_macro
            best_params = params
            best_metrics = {"f1_macro": f1_macro, "f1_binary": f1_binary, "precision": precision, "recall": recall}

    logger.info("==========================================")
    logger.info(f"BEST CONFIGURATION: {best_params}")
    logger.info(f"BEST METRICS: F1-Macro: {best_metrics['f1_macro']:.4f}, Recall (Macro): {best_metrics['recall']:.4f}")
    logger.info("==========================================")

    # Save results to a CSV for easy viewing
    pd.DataFrame(results).to_csv(os.path.join(config.LOG_DIR, "xgb_tuning_results.csv"), index=False)
    logger.info("Results saved to xgb_tuning_results.csv")

if __name__ == "__main__":
    tune_xgboost()
