"""Script for tuning Random Forest hyperparameters."""

import os
import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
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

    logger.info("Splitting training data for validation...")
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
        features_train, labels_train, test_size=0.2, random_state=42, stratify=labels_train
    )

    transformer = build_feature_pipeline(X_train_tune)

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
        
        # Train on tuning set
        pipeline.fit(X_train_tune, y_train_tune)
        
        # Predict on Validation Set
        if hasattr(pipeline, "predict_proba"):
            probas = pipeline.predict_proba(X_val_tune)[:, 1]
            predicted_labels = (probas >= config.DECISION_THRESHOLD).astype(int)
        else:
            predicted_labels = pipeline.predict(X_val_tune)
        
        # Evaluate on Validation Set
        f1_binary = f1_score(y_val_tune, predicted_labels, average="binary")
        f1_macro = f1_score(y_val_tune, predicted_labels, average="macro")
        precision = precision_score(y_val_tune, predicted_labels, average="macro")
        recall = recall_score(y_val_tune, predicted_labels, average="macro")
        
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
    logger.info(f"BEST METRICS (Validation): F1-Macro: {best_metrics['f1_macro']:.4f}, Recall (Macro): {best_metrics['recall']:.4f}")
    logger.info("==========================================")

    # Retrain best model on full training set and evaluate on test set
    logger.info("Retraining best model on full training set...")
    full_transformer = build_feature_pipeline(features_train)
    best_model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", full_transformer),
            ("feature_selection", SelectPercentile(score_func=f_classif, percentile=50)),
            ("classifier", best_model),
        ]
    )
    best_pipeline.fit(features_train, labels_train)

    logger.info("Evaluating best model on Test Set...")
    if hasattr(best_pipeline, "predict_proba"):
        test_probas = best_pipeline.predict_proba(features_test)[:, 1]
        test_predicted = (test_probas >= config.DECISION_THRESHOLD).astype(int)
    else:
        test_predicted = best_pipeline.predict(features_test)
    
    test_f1_macro = f1_score(labels_test, test_predicted, average="macro")
    test_recall = recall_score(labels_test, test_predicted, average="macro")
    test_precision = precision_score(labels_test, test_predicted, average="macro")
    
    logger.info(f"TEST SET RESULTS -> F1-Macro: {test_f1_macro:.4f} | Recall: {test_recall:.4f} | Precision: {test_precision:.4f}")
    logger.info("==========================================")

    # Save results to a CSV for easy viewing
    pd.DataFrame(results).to_csv(os.path.join(config.LOG_DIR, "rf_tuning_results.csv"), index=False)
    logger.info("Results saved to rf_tuning_results.csv")

if __name__ == "__main__":
    tune_random_forest()
