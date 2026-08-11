"""Script to make predictions using saved model pipeline."""

import pandas as pd
from src import config
from src.models.artifacts import ArtifactStore


def predict(input_data: pd.DataFrame, model_path: str = config.MODEL_SAVE_PATH):
    """Load saved pipeline and make predictions on input_data."""
    pipeline = ArtifactStore.load_pipeline(model_path)
    if hasattr(pipeline, "predict_proba"):
        probas = pipeline.predict_proba(input_data)[:, 1]
        return (probas >= config.DECISION_THRESHOLD).astype(int)
    return pipeline.predict(input_data)
