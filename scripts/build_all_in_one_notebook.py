"""สร้าง notebooks/NIDS_All_In_One.ipynb ให้สอดคล้องกับ src/ และ main.py"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "NIDS_All_In_One.ipynb"


def md(text: str) -> dict:
    lines = text.strip().split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in lines],
    }


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

cells.append(md("""# Network intrusion detection — all-in-one notebook

Notebook นี้รัน pipeline เดียวกับ `main.py` โดย **import โค้ดจริงจาก `src/`** เพื่อไม่ให้ซ้ำกับซอร์สหลัก

**ความต้องการ:** `numpy`, `pandas`, `scikit-learn` — ตั้ง working directory เป็นรากโปรเจกต์ (ที่มี `main.py`)

**ข้อมูล:** `data/raw/KDDTrain+.txt`, `data/raw/KDDTest+.txt`"""))

cells.append(code(r"""import sys
from pathlib import Path

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

from src import config
from src.data.preprocessing import IntrusionDatasetPreprocessor, synthesize_fallback_dataset
from src.features.build_features import build_feature_pipeline
from src.models.artifacts import ArtifactStore
from src.models.metrics import BinaryClassifierMetrics
from src.models.model import ModelFactory
from src.visualization.visualize import analyze_feature_importance
from sklearn.pipeline import Pipeline"""))

cells.append(code(r"""logger = ArtifactStore.configure_logger(config.LOG_DIR)
logger.info("Starting intrusion-detection training pipeline (notebook)...")

preprocessor = IntrusionDatasetPreprocessor()

try:
    logger.info(f"Loading training data from {config.DATA_PATH_TRAIN}...")
    features_train, labels_train = preprocessor.load_data(config.DATA_PATH_TRAIN)
    features_test, labels_test = preprocessor.load_data(config.DATA_PATH_TEST)
except FileNotFoundError:
    logger.warning("Training data not found; using synthesized fallback dataset.")
    features_train, labels_train, features_test, labels_test = synthesize_fallback_dataset()

transformer = build_feature_pipeline(features_train)
model = ModelFactory.get_model(config.MODEL_CONFIG)
pipeline = Pipeline(steps=[("preprocessor", transformer), ("classifier", model)])

pipeline.fit(features_train, labels_train)
ArtifactStore.save_pipeline(pipeline, config.MODEL_SAVE_PATH)

if hasattr(pipeline, "predict_proba"):
    probas = pipeline.predict_proba(features_test)[:, 1]
    predicted_labels = (probas >= config.DECISION_THRESHOLD).astype(int)
else:
    predicted_labels = pipeline.predict(features_test)

metrics = BinaryClassifierMetrics(labels_test, predicted_labels)
metrics.log_report(logger=logger)

analyze_feature_importance(
    pipeline.named_steps["classifier"],
    pipeline.named_steps["preprocessor"],
    logger=logger,
)
logger.info("Pipeline finished.")"""))

cells.append(
    md(
        """โค้ดหลักอยู่ในโมดูลย่อยของ `src/` (`src.data`, `src.features`, `src.models`, `src.visualization`) — รีบิลด์ notebook นี้ด้วยคำสั่ง `python scripts/build_all_in_one_notebook.py`"""
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", OUT)
