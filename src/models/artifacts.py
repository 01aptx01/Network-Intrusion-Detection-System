"""บันทึก/โหลดโมเดลด้วย joblib และตั้งค่า logging"""

import logging
import os
from datetime import datetime

import joblib


class ArtifactStore:
    """จัดการไฟล์ log และ pipeline (.joblib)"""

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
    def save_pipeline(pipeline, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(pipeline, filepath)

    @staticmethod
    def load_pipeline(filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline not found at: {filepath}")
        return joblib.load(filepath)
