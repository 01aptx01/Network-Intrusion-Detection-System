"""Script to run 5-Fold Stratified Cross-Validation for specified models."""

import sys
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config
from src.data.preprocessing import IntrusionDatasetPreprocessor, synthesize_fallback_dataset
from src.features.build_features import build_feature_pipeline
from src.models.artifacts import ArtifactStore
from src.models.model import ModelFactory


def run_cross_validation(models_to_test=["lightgbm", "catboost", "xgboost", "random_forest"], n_splits=5):
    """Run cross-validation on specified models.
    
    Args:
        models_to_test (list): List of model types to test (keys in MODEL_CONFIG['params']).
        n_splits (int): Number of folds for cross-validation.
    """
    logger = ArtifactStore.configure_logger(config.LOG_DIR)
    logger.info(f"Starting {n_splits}-Fold Stratified Cross Validation pipeline...")

    preprocessor = IntrusionDatasetPreprocessor()

    try:
        logger.info(f"Loading data from {config.DATA_PATH_TRAIN} for Cross-Validation...")
        # For CV, we generally only use the training set to avoid touching the test set.
        X, y = preprocessor.load_data(config.DATA_PATH_TRAIN)
    except FileNotFoundError:
        logger.warning("Training data not found; using synthesized fallback dataset.")
        X, y, _, _ = synthesize_fallback_dataset()
        
    logger.info("Creating preprocessor transformer...")
    transformer = build_feature_pipeline(X)
    
    cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    results = {}

    for model_type in models_to_test:
        logger.info(f"\n{'='*50}\nEvaluating Model: {model_type.upper()}\n{'='*50}")
        
        # Prepare specific config for this model
        current_config = config.MODEL_CONFIG.copy()
        current_config["type"] = model_type
        
        model = ModelFactory.get_model(current_config)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", transformer),
                ("feature_selection", SelectPercentile(score_func=f_classif, percentile=50)),
                ("classifier", model),
            ]
        )

        logger.info(f"Running Cross-Validation for {model_type}...")
        
        # We use n_jobs=1 for cross_validate if the model itself uses n_jobs=-1 to avoid nested parallelism issues
        # LightGBM and RandomForest both use n_jobs=-1 by default in our config.
        cv_results = cross_validate(
            pipeline, X, y, 
            cv=cv_strategy, 
            scoring=scoring_metrics, 
            n_jobs=1,
            return_train_score=False
        )
        
        results[model_type] = cv_results
        
        # Print metrics
        logger.info(f"--- Results for {model_type} ---")
        for metric in scoring_metrics:
            scores = cv_results[f"test_{metric}"]
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.info(f"{metric.capitalize():>10}: {mean_score:.4f} (+/- {std_score:.4f})")
            
        logger.info(f"Fit times: {np.mean(cv_results['fit_time']):.2f}s (+/- {np.std(cv_results['fit_time']):.2f}s)")
        logger.info(f"Score times: {np.mean(cv_results['score_time']):.2f}s (+/- {np.std(cv_results['score_time']):.2f}s)")

    logger.info("\nCross-Validation Finished.")
    return results


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, AttributeError):
                pass
    run_cross_validation(models_to_test=["lightgbm", "catboost", "xgboost", "random_forest"])
