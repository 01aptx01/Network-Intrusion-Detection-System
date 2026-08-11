"""Data preprocessor for NSL-KDD dataset."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class IntrusionDatasetPreprocessor:
    """โหลด NSL-KDD และสร้าง scikit-learn preprocessor"""

    def load_data(self, filepath: str) -> tuple:
        try:
            frame = pd.read_csv(filepath, header=None)
        except FileNotFoundError:
            raise FileNotFoundError(f"ไม่พบไฟล์ข้อมูลที่: {filepath}")

        if frame.shape[1] >= 43:
            frame = frame.drop(columns=[42])

        label_column_index = 41
        labels = np.where(frame[label_column_index] == "normal", 0, 1)
        features_frame = frame.drop(columns=[label_column_index])

        return features_frame, labels

    def create_preprocessor(self, features_frame: pd.DataFrame) -> ColumnTransformer:
        """สร้างและคืนค่า scikit-learn ColumnTransformer"""
        categorical_cols = features_frame.select_dtypes(
            include=["object", "string"]
        ).columns
        numeric_cols = features_frame.select_dtypes(
            exclude=["object", "string"]
        ).columns

        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )
        return preprocessor


def synthesize_fallback_dataset() -> tuple:
    """สร้างข้อมูลสุ่มเมื่อยังไม่มีไฟล์ NSL-KDD ใน data/raw"""
    np.random.seed(42)
    features_train = pd.DataFrame(np.random.randn(5000, 41) * 50)
    features_train[1] = np.random.choice(["tcp", "udp", "icmp"], size=5000)
    features_train[2] = np.random.choice(["http", "ftp", "smtp"], size=5000)
    features_train[3] = np.random.choice(["SF", "S0", "REJ"], size=5000)
    labels_train = np.random.choice([0, 1], size=5000, p=[0.8, 0.2])

    features_test = pd.DataFrame(np.random.randn(1000, 41) * 50)
    features_test[1] = np.random.choice(["tcp", "udp", "icmp"], size=1000)
    features_test[2] = np.random.choice(["http", "ftp", "smtp"], size=1000)
    features_test[3] = np.random.choice(["SF", "S0", "REJ"], size=1000)
    labels_test = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])

    features_train.columns = list(range(41))
    features_test.columns = list(range(41))

    return features_train, labels_train, features_test, labels_test
