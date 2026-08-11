# Network Intrusion Detection System - Architecture Guide

This document provides a high-level overview of the system architecture and the workflow of the Network Intrusion Detection System (NIDS).

## 1. System Overview

This project is structured based on the **Cookiecutter Data Science** methodology. The main goal is to load network traffic data (NSL-KDD dataset), preprocess it (e.g., standardizing numerical features and one-hot encoding categorical features), and train a binary classifier (normal vs. attack) using Scikit-Learn.

## 2. Directory Structure (`src/`)

The core source code of the project resides in the `src/` directory, broken down by logical function:

*   **`src/config.py`**: The central configuration file. It stores file paths (e.g., to the raw dataset and saved models) and hyperparameters (e.g., learning rate, decision threshold, and model configurations like `C` for Logistic Regression or `n_estimators` for Random Forest).
*   **`src/data/preprocessing.py`**: Handles loading the data and configuring the `scikit-learn` `ColumnTransformer` (which applies `StandardScaler` to numerical data and `OneHotEncoder` to categorical data). It also contains a fallback function to synthesize fake data if the actual NSL-KDD dataset is missing.
*   **`src/features/build_features.py`**: Provides helper functions to connect the raw preprocessing logic into the training pipeline.
*   **`src/models/`**: Contains the logic for defining, training, saving, and evaluating models.
    *   **`model.py`**: Defines a `ModelFactory` that instantiates the correct Scikit-Learn model based on the configuration in `config.py`.
    *   **`train_model.py`**: The main training script that pieces everything together (Data Loading -> Preprocessing -> Training -> Evaluation).
    *   **`predict_model.py`**: Functions for making predictions using a pre-trained and saved model pipeline.
    *   **`metrics.py`**: Custom classes for calculating and logging classification metrics (Precision, Recall, F1-score, Confusion Matrix).
    *   **`artifacts.py`**: Helper functions for saving/loading the trained models (`.joblib` files) and configuring the logging system.
*   **`src/visualization/visualize.py`**: Logic for generating plots, such as the feature importance chart, based on the trained model.

## 3. Pipeline Workflow

When you execute `python main.py`, the following sequence of events occurs:

1.  **Entry Point (`main.py`)**: The script calls the `train_pipeline()` function from `src.models.train_model`.
2.  **Configuration & Logging**: `train_pipeline` first initializes the logger (saving logs to `logs/`) and reads settings from `src/config.py`.
3.  **Data Loading (`src.data.preprocessing`)**: It attempts to load `KDDTrain+.txt` and `KDDTest+.txt`. If these are not found, it generates a synthetic dataset to ensure the pipeline doesn't crash during testing.
4.  **Pipeline Construction (`scikit-learn Pipeline`)**: 
    *   A `ColumnTransformer` is created for preprocessing.
    *   A model is instantiated via `ModelFactory` (e.g., Logistic Regression).
    *   These are combined into a single `scikit-learn` `Pipeline`.
5.  **Training**: `pipeline.fit(features_train, labels_train)` is called. The data flows through the preprocessor and then trains the classifier.
6.  **Saving Artifacts (`src.models.artifacts`)**: The entire pipeline (preprocessor + model) is serialized and saved to `models/nids_pipeline.joblib`.
7.  **Evaluation (`src.models.metrics`)**: The pipeline predicts on the test set. The predictions are evaluated against the true labels, generating metrics like True Positives, False Positives, Precision, and Recall.
8.  **Visualization (`src.visualization.visualize`)**: Feature importance is extracted from the model and plotted, saving the output to `docs/images/`.

This pipeline architecture ensures that preprocessing steps applied during training are identically applied during inference, preventing data leakage and deployment bugs.
