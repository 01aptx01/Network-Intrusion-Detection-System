import os
import logging
import numpy as np
from datetime import datetime

class NIDSUtils:
    @staticmethod
    def setup_logger(log_dir: str = "logs") -> logging.Logger:
        """
        สร้างระบบ Logging เพื่อบันทึก Loss และ Metrics ลงไฟล์
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # สร้างชื่อไฟล์ตามเวลาปัจจุบัน
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"nids_training_{timestamp}.log")

        logger = logging.getLogger("NIDS_Logger")
        logger.setLevel(logging.INFO)

        # ป้องกันการสร้าง Handler ซ้ำซ้อนถ้าเรียกฟังก์ชันนี้หลายรอบ
        if not logger.handlers:
            # File Handler
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            
            # Console Handler (แสดงบน Terminal ด้วย)
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            # Format
            formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    @staticmethod
    def save_model_weights(W: np.ndarray, b: float, filepath: str = "saved_models/nids_weights.npz"):
        """
        บันทึก Weight Matrix และ Bias ลงในไฟล์ Binary (.npz) เพื่อความเร็วในการโหลด
        Time Complexity: O(M) - เขียนอาเรย์ขนาดเท่ากับจำนวน Features ลงดิสก์
        Space Complexity: O(M) - ขนาดไฟล์บนดิสก์แปรผันตามจำนวน Features
        """
        # ตรวจสอบและสร้างโฟลเดอร์ถ้ายังไม่มี
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # np.savez เก็บหลายๆ Arrays ลงในไฟล์เดียว
        np.savez(filepath, W=W, b=np.array([b]))
        print(f"[*] Model weights successfully saved to {filepath}")

    @staticmethod
    def load_model_weights(filepath: str = "saved_models/nids_weights.npz") -> tuple:
        """
        โหลด Weight Matrix และ Bias จากไฟล์ Binary กลับเข้าสู่ RAM
        Time Complexity: O(M)
        Space Complexity: O(M) - จอง RAM คืนตามขนาดของ Weights
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"ไม่พบไฟล์ Weights ที่: {filepath}")
            
        with np.load(filepath) as data:
            W = data['W']
            b = data['b'][0] # ดึง scalar ออกจาก array 1D
            
        print(f"[*] Model weights loaded from {filepath}")
        return W, b