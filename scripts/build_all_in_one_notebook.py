"""สร้าง notebooks/NIDS_All_In_One.ipynb ให้สอดคล้องกับ src/ และ main.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "NIDS_All_In_One.ipynb"


def md(text: str) -> dict:
    lines = text.strip().split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def code(text: str) -> dict:
    text = text.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


cells = []

cells.append(
    md(
        """# Network intrusion detection — all-in-one notebook

Notebook นี้รัน pipeline เดียวกับ `main.py` โดย **import โค้ดจริงจาก `src/`** เพื่อไม่ให้ซ้ำกับซอร์สหลัก

**ความต้องการ:** `numpy`, `pandas` — ตั้ง working directory เป็นรากโปรเจกต์ (ที่มี `main.py`) หรือโฟลเดอร์ `notebooks/` (สคริปต์จะหา `BASE_DIR` ให้)

**ข้อมูล:** `data/raw/KDDTrain+.txt`, `data/raw/KDDTest+.txt`"""
    )
)

cells.append(
    code(
        r'''import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (OSError, AttributeError):
        pass


def project_root() -> Path:
    p = Path.cwd().resolve()
    for _ in range(6):
        if (p / "main.py").exists() and (p / "src").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return Path.cwd().resolve()


BASE_DIR = project_root()
sys.path.insert(0, str(BASE_DIR))
import os

os.chdir(BASE_DIR)
print("BASE_DIR =", BASE_DIR)

spec = importlib.util.spec_from_file_location("config", BASE_DIR / "config.py")
config = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(config)

from src.artifacts import ArtifactStore
from src.metrics import BinaryClassifierMetrics
from src.model import BinaryLogisticRegression
from src.preprocessing import IntrusionDatasetPreprocessor'''
    )
)

cells.append(
    code(
        r'''def synthesize_fallback_dataset():
    np.random.seed(42)
    features_train = np.random.randn(5000, 41) * 50
    labels_train = np.random.choice([0, 1], size=5000, p=[0.8, 0.2])
    features_test = np.random.randn(1000, 41) * 50
    labels_test = np.random.choice([0, 1], size=1000, p=[0.8, 0.2])
    return features_train, labels_train, features_test, labels_test


def analyze_feature_importance(weights: np.ndarray, feature_names: pd.Index, logger=None):
    weights_flat = weights.flatten()
    sorted_indices = np.argsort(weights_flat)
    lines = ["\n" + "=" * 60, "Feature importance (linear weights)", "=" * 60]
    top_attack_idx = sorted_indices[-5:][::-1]
    lines.append("Top 5 attack indicators (most positive weights):")
    for i, idx in enumerate(top_attack_idx):
        lines.append(f"   {i + 1}. {str(feature_names[idx]):<30} : {weights_flat[idx]:+.4f}")
    lines.append("-" * 60)
    top_normal_idx = sorted_indices[:5]
    lines.append("Top 5 normal indicators (most negative weights):")
    for i, idx in enumerate(top_normal_idx):
        lines.append(f"   {i + 1}. {str(feature_names[idx]):<30} : {weights_flat[idx]:+.4f}")
    lines.append("=" * 60)
    output = "\n".join(lines)
    if logger:
        logger.info(output)
    else:
        print(output)'''
    )
)

cells.append(
    code(
        r'''logger = ArtifactStore.configure_logger(config.LOG_DIR)
logger.info("Starting intrusion-detection training pipeline (notebook)...")

preprocessor = IntrusionDatasetPreprocessor()

try:
    logger.info(f"Loading training data from {config.DATA_PATH_TRAIN}...")
    features_train, labels_train = preprocessor.load_data(config.DATA_PATH_TRAIN, is_train=True)
    features_test, labels_test = preprocessor.load_data(config.DATA_PATH_TEST, is_train=False)
except FileNotFoundError:
    logger.warning("Training data not found; using synthesized fallback dataset.")
    features_train, labels_train, features_test, labels_test = synthesize_fallback_dataset()

logger.info("Applying Z-score normalization...")
features_train_scaled = preprocessor.fit_transform(features_train)
features_test_scaled = preprocessor.transform(features_test)

class_weights = preprocessor.compute_inverse_frequency_class_weights(labels_train)
attack_loss_weight = class_weights[1] / class_weights[0]
logger.info(f"Class imbalance factor (attack vs normal): {attack_loss_weight:.2f}x")

model = BinaryLogisticRegression(
    learning_rate=config.LEARNING_RATE,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
)
model.fit(
    features_train_scaled,
    labels_train,
    attack_loss_weight=attack_loss_weight,
    logger=logger,
)

ArtifactStore.save_model_weights(model.weights, model.bias, config.MODEL_SAVE_PATH)
prep_path = config.MODEL_SAVE_PATH.replace(".npz", "_preprocessor.pkl")
ArtifactStore.save_preprocessor(preprocessor, prep_path)

predicted_labels = model.predict(
    features_test_scaled, decision_threshold=config.DECISION_THRESHOLD
)
metrics = BinaryClassifierMetrics(labels_test, predicted_labels)
metrics.log_report(logger=logger)

analyze_feature_importance(model.weights, preprocessor.train_column_names, logger=logger)
logger.info("Pipeline finished.")'''
    )
)

cells.append(
    md(
        """โค้ดหลักอยู่ที่ `src/artifacts.py`, `src/preprocessing.py`, `src/model.py`, `src/metrics.py` และ `main.py` — รีบิลด์ notebook นี้ด้วยคำสั่ง `python scripts/build_all_in_one_notebook.py`"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT)
