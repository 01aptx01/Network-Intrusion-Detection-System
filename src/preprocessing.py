import numpy as np
import pandas as pd


class IntrusionDatasetPreprocessor:
    """โหลด NSL-KDD แบบ comma-separated, one-hot categorical, Z-score จากชุดฝึก"""

    def __init__(self):
        self.feature_mean = None
        self.feature_std = None
        self.is_fitted = False
        self.train_column_names = None

    def load_data(self, filepath: str, is_train: bool = True) -> tuple:
        try:
            frame = pd.read_csv(filepath, header=None)
        except FileNotFoundError:
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลที่: {filepath}")

        if frame.shape[1] >= 43:
            frame = frame.drop(columns=[42])

        label_column_index = 41
        labels = np.where(frame[label_column_index] == "normal", 0, 1)
        features_frame = frame.drop(columns=[label_column_index])
        features_frame = pd.get_dummies(features_frame)

        if is_train:
            self.train_column_names = features_frame.columns
        else:
            features_frame = features_frame.reindex(columns=self.train_column_names, fill_value=0)

        return features_frame.values.astype(np.float64), labels

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0)
        self.feature_std[self.feature_std == 0] = 1e-8
        self.is_fitted = True
        return (features - self.feature_mean) / self.feature_std

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ต้อง fit_transform บนชุดฝึกก่อน")
        return (features - self.feature_mean) / self.feature_std

    @staticmethod
    def compute_inverse_frequency_class_weights(labels: np.ndarray) -> dict:
        num_samples = len(labels)
        classes, counts = np.unique(labels, return_counts=True)
        return {int(cls): num_samples / (len(classes) * count) for cls, count in zip(classes, counts)}
