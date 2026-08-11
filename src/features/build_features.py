"""Scripts to turn raw data into features for modeling."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from src.data.preprocessing import IntrusionDatasetPreprocessor


def build_feature_pipeline(features_frame: pd.DataFrame) -> ColumnTransformer:
    """Build feature transformer pipeline for NSL-KDD dataset."""
    preprocessor = IntrusionDatasetPreprocessor()
    return preprocessor.create_preprocessor(features_frame)
