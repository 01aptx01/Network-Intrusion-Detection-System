"""Unit test for intrusion detection pipeline."""

import pytest
from src.data.preprocessing import IntrusionDatasetPreprocessor, synthesize_fallback_dataset
from src.features.build_features import build_feature_pipeline
from src.models.model import ModelFactory
from src import config


def test_fallback_dataset_synthesis():
    X_train, y_train, X_test, y_test = synthesize_fallback_dataset()
    assert len(X_train) == 5000
    assert len(X_test) == 1000
    assert len(y_train) == 5000
    assert len(y_test) == 1000


def test_model_factory_logistic_regression():
    model = ModelFactory.get_model(config.MODEL_CONFIG)
    assert model is not None


def test_feature_pipeline_creation():
    X_train, _, _, _ = synthesize_fallback_dataset()
    transformer = build_feature_pipeline(X_train)
    assert transformer is not None
