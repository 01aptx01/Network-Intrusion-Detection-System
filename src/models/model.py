"""Model factory for creating classifier instances."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class ModelFactory:
    """Factory สำหรับสร้าง scikit-learn models ตาม configuration"""

    @staticmethod
    def get_model(config_dict: dict):
        model_type = config_dict.get("type", "logistic_regression")
        params = config_dict.get("params", {}).get(model_type, {})

        if model_type == "logistic_regression":
            return LogisticRegression(**params)
        elif model_type == "random_forest":
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
