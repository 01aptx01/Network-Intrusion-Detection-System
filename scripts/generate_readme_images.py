"""
สร้างภาพประกอบ README (รันแยกจาก pipeline หลัก — ใช้ matplotlib)
Usage: pip install matplotlib && python scripts/generate_readme_images.py
"""

from __future__ import annotations

import os
import sys

# รากโปรเจกต์
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "images")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) สรุปผลจากรัน pipeline จริง (NSL-KDD KDDTest+.txt, threshold=0.4, config ปัจจุบัน)
    labels = ("TP\n(Caught)", "TN\n(Normal)", "FP\n(False alarm)", "FN\n(Missed)")
    values = (8580, 9422, 289, 4253)
    colors = ("#2ecc71", "#3498db", "#f39c12", "#e74c3c")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#2c3e50", linewidth=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Binary classifier evaluation — confusion components (KDDTest+)")
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 80,
            str(v),
            ha="center",
            fontsize=11,
        )
    ax.set_ylim(0, max(values) * 1.12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "evaluation_counts.png"), dpi=150)
    plt.close(fig)

    # 2) Pipeline flow (ไม่ใช่ screenshot จอจริง แต่เป็นไดอะแกรมภาพรวม)
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")
    boxes = [
        "Load NSL-KDD\n(train/test)",
        "Preprocess\n(Encode/Scale)",
        "Feature Select\n(Percentile=50)",
        "XGBoost / Tree\nClassifier",
        "Threshold +\nMetrics Eval",
    ]
    x = [0, 2, 4, 6, 8]
    y = [0.5] * 5
    for i, (xi, yi, t) in enumerate(zip(x, y, boxes)):
        ax.add_patch(
            plt.Rectangle(
                (xi - 0.75, yi - 0.35),
                1.5,
                0.7,
                fill=True,
                facecolor="#ecf0f1",
                edgecolor="#34495e",
                linewidth=1.2,
            )
        )
        ax.text(xi, yi, t, ha="center", va="center", fontsize=9)
        if i < len(x) - 1:
            ax.annotate(
                "",
                xy=(x[i + 1] - 0.75, yi),
                xytext=(xi + 0.75, yi),
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5),
            )
    ax.set_xlim(-1.2, 9.2)
    ax.set_ylim(0, 1)
    ax.set_title("End-to-end pipeline (main.py)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(
        os.path.join(OUT_DIR, "pipeline_overview.png"), dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Wrote images to {OUT_DIR}")


if __name__ == "__main__":
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Install matplotlib: pip install matplotlib", file=sys.stderr)
        sys.exit(1)
    main()
