import os
import logging
import numpy as np
from datetime import datetime

class NIDSUtils:
    @staticmethod
    def setup_logger(log_dir: str) -> logging.Logger:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"nids_training_{timestamp}.log")

        logger = logging.getLogger("NIDS_Logger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    @staticmethod
    def save_model(W: np.ndarray, b: float, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath, W=W, b=np.array([b]))

    @staticmethod
    def load_model(filepath: str) -> tuple:
        with np.load(filepath) as data:
            return data['W'], data['b'][0]