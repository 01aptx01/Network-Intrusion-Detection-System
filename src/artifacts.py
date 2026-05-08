"""บันทึก/โหลดน้ำหนักโมเดล preprocessor และตั้งค่า logging สำหรับการฝึก"""
import logging
import os
import pickle
from datetime import datetime

import numpy as np


class ArtifactStore:
    """จัดการไฟล์ log, weights (.npz) และ preprocessor (.pkl)"""

    _LOGGER_NAME = "intrusion_detection.training"

    @staticmethod
    def configure_logger(log_dir: str) -> logging.Logger:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_run_{timestamp}.log")

        logger = logging.getLogger(ArtifactStore._LOGGER_NAME)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

        return logger

    @staticmethod
    def save_model_weights(weights: np.ndarray, bias: float, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, W=weights, b=np.array([bias]))

    @staticmethod
    def load_model_weights(filepath: str) -> tuple:
        with np.load(filepath) as data:
            return data["W"], float(data["b"][0])

    @staticmethod
    def save_preprocessor(preprocessor, filepath: str = "saved_models/preprocessor.pkl") -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file_obj:
            pickle.dump(preprocessor, file_obj)
        print(f"[*] Preprocessor saved to {filepath}")

    @staticmethod
    def load_preprocessor(filepath: str = "saved_models/preprocessor.pkl"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ไม่พบไฟล์ Preprocessor ที่: {filepath}")
        with open(filepath, "rb") as file_obj:
            return pickle.load(file_obj)
