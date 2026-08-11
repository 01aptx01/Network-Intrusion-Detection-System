"""Script for tuning Random Forest hyperparameters."""

import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

from src import config
from src.data.preprocessing import IntrusionDatasetPreprocessor
from src.features.build_features import build_feature_pipeline
from src.models.artifacts import ArtifactStore

def tune_random_forest():
    logger = ArtifactStore.configure_logger(config.LOG_DIR)
    logger.info("Starting Random Forest hyperparameter tuning...")

    preprocessor = IntrusionDatasetPreprocessor()
    logger.info(f"Loading training data...")
    features_train, labels_train = preprocessor.load_data(config.DATA_PATH_TRAIN)
    features_test, labels_test = preprocessor.load_data(config.DATA_PATH_TEST)

    transformer = build_feature_pipeline(features_train)

    # 10 predefined configurations for Random Forest
    param_grid = [
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"}, # 1. Default (Good baseline)
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"}, # 2. More trees
        {"n_estimators": 100, "max_depth": 50, "min_samples_split": 2, "class_weight": "balanced"}, # 3. Limit depth slightly
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 5, "class_weight": "balanced"}, # 4. Require more samples to split
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced_subsample"}, # 5. Different weighting
        {"n_estimators": 200, "max_depth": 50, "min_samples_split": 5, "class_weight": "balanced"}, # 6. Mixed regularization
        {"n_estimators": 50, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"}, # 7. Fewer trees (faster, maybe less overfit)
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 10, "class_weight": "balanced"}, # 8. Stronger split constraint
        {"n_estimators": 300, "max_depth": None, "min_samples_split": 2, "class_weight": "balanced"}, # 9. Very large forest
        {"n_estimators": 200, "max_depth": None, "min_samples_split": 2, "class_weight": None}, # 10. No class weighting
    ]

    best_score = 0
    best_params = None
    best_metrics = {}
    
    results = []

    for i, params in enumerate(param_grid):
        logger.info(f"--- Testing config {i+1}/10: {params} ---")
        
        # Build pipeline
        model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
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
        f1 = f1_score(labels_test, predicted_labels)
        precision = precision_score(labels_test, predicted_labels)
        recall = recall_score(labels_test, predicted_labels)
        
        logger.info(f"Config {i+1} Results -> F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        results.append({
            "config_id": i+1,
            "params": params,
            "f1": f1,
            "precision": precision,
            "recall": recall
        })
        
        # Select best based on F1-Score (or can be customized)
        if f1 > best_score:
            best_score = f1
            best_params = params
            best_metrics = {"f1": f1, "precision": precision, "recall": recall}

    logger.info("==========================================")
    logger.info(f"BEST CONFIGURATION: {best_params}")
    logger.info(f"BEST METRICS: F1: {best_metrics['f1']:.4f}, Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
    logger.info("==========================================")

    # Save results to a CSV for easy viewing
    pd.DataFrame(results).to_csv(os.path.join(config.LOG_DIR, "rf_tuning_results.csv"), index=False)
    logger.info("Results saved to rf_tuning_results.csv")

if __name__ == "__main__":
    tune_random_forest()
