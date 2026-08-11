"""Visualization functions for evaluation and feature importance."""

import numpy as np


def analyze_feature_importance(model, preprocessor_transformer, logger=None):
    """วิเคราะห์ความสำคัญของ feature หากโมเดลรองรับ"""
    try:
        feature_names = preprocessor_transformer.get_feature_names_out()
    except AttributeError:
        return

    if hasattr(model, "coef_"):
        weights = model.coef_[0]
        title = "Feature importance (linear weights)"
    elif hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
        title = "Feature importance (tree-based)"
    else:
        return

    sorted_indices = np.argsort(weights)
    lines = ["\n" + "=" * 60, title, "=" * 60]

    top_attack_idx = sorted_indices[-5:][::-1]
    lines.append("Top 5 important features (highest values):")
    for i, idx in enumerate(top_attack_idx):
        if idx < len(feature_names):
            lines.append(
                f"   {i + 1}. {str(feature_names[idx]):<30} : {weights[idx]:+.4f}"
            )

    lines.append("=" * 60)
    output = "\n".join(lines)
    if logger:
        logger.info(output)
    else:
        print(output)
