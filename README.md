# Network Intrusion Detection System (NIDS)

<div align="center">
  <p>A binary network traffic classification system to detect normal behaviors and cyber-attacks.</p>
</div>

## 📌 Project Overview
This project implements a Network Intrusion Detection System (NIDS) as a binary classifier (normal vs. attack) using the **NSL-KDD** dataset. The core model focuses on **Logistic Regression** and robust data pipelines, utilizing standard machine learning frameworks like **Scikit-learn** to ensure production-ready quality and maintainability.

The project structure adheres to the industry-standard **Cookiecutter Data Science** methodology to ensure reproducibility, scalability, and maintainability.

## 🚀 Features
- **Model Implementations**: Leveraging robust libraries like Scikit-learn alongside custom implementations.
- **Robust Preprocessing**: Handles categorical variable one-hot encoding and Z-score normalization based on training statistics.
- **Class Imbalance Handling**: Automatically calculates and applies class weights during model training.
- **Comprehensive Evaluation**: Generates confusion matrix components, precision, recall, F1-score, and feature importance.
- **Cookiecutter Architecture**: Well-organized file and directory structure following global best practices for Data Science.

## 📂 Project Organization

```text
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump (e.g., KDDTrain+.txt, KDDTest+.txt).
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│   └── images         <- Project images and visualizations
│
├── logs               <- Execution logs
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── scripts            <- Standalone scripts (e.g., generate_readme_images.py, build_all_in_one_notebook.py)
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── config.py      <- Configuration variables (paths, hyperparameters)
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── preprocessing.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make predictions
│   │   ├── artifacts.py
│   │   ├── metrics.py
│   │   ├── model.py
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tests              <- Unit tests for the pipeline
```

## 📊 Visualizations

| Pipeline Overview | Evaluation Summary |
| :---: | :---: |
| ![Pipeline overview](docs/images/pipeline_overview.png) | ![Evaluation counts](docs/images/evaluation_counts.png) |

> **Note**: The red bars (FN - missed attacks) provide immediate visual feedback on the limitations of thresholding and linear models in real-world scenarios.

## ⚙️ Requirements & Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Network-Intrusion-Detection-System

# Install dependencies
pip install -r requirements.txt
```

> Ensure you have the NSL-KDD raw data files `KDDTrain+.txt` and `KDDTest+.txt` placed in the `data/raw/` directory.

## 💻 Usage

To run the full pipeline (load, preprocess, train, evaluate, and save):

```bash
python main.py
```

*   **Logs**: Check the `logs/` directory for detailed execution logs.
*   **Model Weights**: Saved in the `models/` directory (e.g., `.npz` weights and `.pkl` preprocessor states).

To generate README images (requires `matplotlib`):
```bash
pip install matplotlib
python scripts/generate_readme_images.py
```

## 🧪 Evaluation Metrics

Reference results from running `main.py` on the **KDDTrain+.txt** and **KDDTest+.txt** datasets using standard hyperparameters (`LEARNING_RATE=0.05`, `EPOCHS=50`, `BATCH_SIZE=128`, `DECISION_THRESHOLD=0.4`):

| Metric | Value |
|--------|------:|
| **True Positives (caught attacks)** | 8,088 |
| **True Negatives (normal)** | 8,943 |
| **False Positives (false alarms)** | 768 |
| **False Negatives (missed attacks)** | 4,745 |
| **Precision** | 0.9133 |
| **Recall** | 0.6303 |
| **F1-score** | 0.7458 |

*Note: Results may vary based on hyperparameter tuning, thresholds, and mini-batch sampling.*

## 🛠️ Technology Stack

- **Language:** Python 3.x
- **Core Libraries:** `numpy`, `pandas`, `scikit-learn`

## 📝 Commit Guidelines

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification.
Examples:
- `feat(model): add class weights`
- `docs(readme): update project structure`
- `fix(metrics): resolve division by zero in recall`
- `refactor(src): adopt cookiecutter data science standard`
