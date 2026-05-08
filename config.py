"""
Configuration File
เก็บพารามิเตอร์ทั้งหมดไว้ที่เดียวเพื่อง่ายต่อการปรับแต่ง (Tuning)
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# NSL-KDD มักจัดเก็บเป็น .txt (comma-separated); read_csv อ่านได้เหมือน CSV
DATA_PATH_TRAIN = os.path.join(BASE_DIR, "data/raw/KDDTrain+.txt")
DATA_PATH_TEST = os.path.join(BASE_DIR, "data/raw/KDDTest+.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models/nids_weights.npz")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Hyperparameters สำหรับ Mini-Batch Gradient Descent
LEARNING_RATE = 0.05
EPOCHS = 50
BATCH_SIZE = 128
DECISION_THRESHOLD = 0.4  # ปรับลดลงเพื่อลด False Negative