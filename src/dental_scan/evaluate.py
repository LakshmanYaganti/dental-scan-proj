"""Evaluate a trained model on a held-out folder-per-class dataset.

Usage::

    PYTHONPATH=src python -m dental_scan.evaluate \
        --config configs/config.yaml \
        --model models/dental_scan.keras \
        --data-dir data/test
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from dental_scan.data import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the dental scan classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default="models/dental_scan.keras")
    parser.add_argument("--data-dir", default=None, help="Defaults to data.test_dir in config.")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    import tensorflow as tf

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    data_cfg = cfg.get("data", {})
    image_size = tuple(data_cfg.get("image_size", [224, 224]))
    batch_size = int(data_cfg.get("batch_size", 32))
    data_dir = args.data_dir or data_cfg.get("test_dir", "data/test")

    dataset, class_names = load_dataset(
        data_dir, image_size=image_size, batch_size=batch_size, shuffle=False
    )

    model = tf.keras.models.load_model(args.model)

    y_true: list[int] = []
    for _, labels in dataset:
        y_true.extend(labels.numpy().tolist())
    y_prob = model.predict(dataset)
    y_pred = np.argmax(y_prob, axis=1).tolist()

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    print(report)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))

    out = pathlib.Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classification_report.txt").write_text(report, encoding="utf-8")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="left")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Dental Scan — Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out / "confusion_matrix.png", dpi=150)
        print(f"Wrote {out / 'confusion_matrix.png'}")
    except ImportError:
        print("matplotlib not installed — skipped confusion matrix plot.")


if __name__ == "__main__":
    main()
