"""
Configuration File
เก็บพารามิเตอร์ทั้งหมดไว้ที่เดียวเพื่อง่ายต่อการปรับแต่ง (Tuning)
"""

import os

# Project root directory (parent directory of src/)
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE_DIR = PROJECT_DIR  # Backward compatibility alias

# Paths
DATA_PATH_TRAIN = os.path.join(PROJECT_DIR, "data", "raw", "KDDTrain+.txt")
DATA_PATH_TEST = os.path.join(PROJECT_DIR, "data", "raw", "KDDTest+.txt")
MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, "models", "nids_pipeline.joblib")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")

# Hyperparameters
DECISION_THRESHOLD = 0.4  # ปรับลดลงเพื่อลด False Negative

# Model Configuration (Scikit-Learn)
MODEL_CONFIG = {
    "type": "lightgbm",  # Options: "logistic_regression", "random_forest", "lightgbm", "xgboost", "catboost"
    "params": {
        "logistic_regression": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
        },
        "random_forest": {
            "n_estimators": 100,
            "class_weight": "balanced",
            "max_depth": None,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,
        },
        "lightgbm": {
            "n_estimators": 300,
            "max_depth": -1,
            "learning_rate": 0.01,
            "scale_pos_weight": 10,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "xgboost": {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.01,
            "scale_pos_weight": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "catboost": {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.01,
            "scale_pos_weight": 10,
            "random_state": 42,
            "verbose": False,
        },
    },
}
