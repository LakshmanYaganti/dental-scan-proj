"""CLI training entrypoint.

Usage::

    PYTHONPATH=src python -m dental_scan.train --config configs/config.yaml

MLflow logging is optional: if ``mlflow`` is not installed, training
proceeds without it and prints a one-line notice.
"""
from __future__ import annotations

import argparse
import json
import pathlib

import yaml

from dental_scan.data import load_dataset, load_train_val
from dental_scan.model import build_model


def load_config(path: str | pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dental scan classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    mlflow_cfg = cfg.get("mlflow", {})

    image_size = tuple(data_cfg.get("image_size", [224, 224]))
    batch_size = int(data_cfg.get("batch_size", 32))
    seed = int(data_cfg.get("seed", 42))
    epochs = args.epochs or int(train_cfg.get("epochs", 15))

    train_dir = pathlib.Path(data_cfg.get("train_dir", "data/train"))
    val_dir = data_cfg.get("val_dir")

    if val_dir and pathlib.Path(val_dir).is_dir():
        train_ds, class_names = load_dataset(
            train_dir, image_size=image_size, batch_size=batch_size, seed=seed, augment=True
        )
        val_ds, _ = load_dataset(
            val_dir, image_size=image_size, batch_size=batch_size, seed=seed, shuffle=False
        )
    else:
        train_ds, val_ds, class_names = load_train_val(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            validation_split=float(data_cfg.get("validation_split", 0.15)),
            seed=seed,
        )

    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(
        num_classes=len(class_names),
        image_size=image_size,
        backbone=model_cfg.get("backbone", "efficientnetb0"),
        dropout=float(model_cfg.get("dropout", 0.3)),
        learning_rate=float(model_cfg.get("learning_rate", 1e-3)),
    )
    model.summary()

    # Optional MLflow — guarded so training never depends on it.
    mlflow_run = None
    if mlflow_cfg.get("enabled", False):
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
            mlflow.set_experiment(mlflow_cfg.get("experiment_name", "dental-scan-proj"))
            mlflow_run = mlflow.start_run()
            mlflow.log_params(
                {
                    "backbone": model_cfg.get("backbone", "efficientnetb0"),
                    "image_size": list(image_size),
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "num_classes": len(class_names),
                    "classes": ",".join(class_names),
                }
            )
        except ImportError:
            print("mlflow not installed — continuing without experiment tracking.")

    callbacks = [
        # Save time on plateaus; keep the run reproducible.
        __import__("tensorflow").keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        __import__("tensorflow").keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
    )

    output_model = pathlib.Path(train_cfg.get("output_model", "models/dental_scan.keras"))
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model)
    print(f"Saved model -> {output_model}")

    labels_path = pathlib.Path(train_cfg.get("class_labels", "models/class_labels.json"))
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved class labels -> {labels_path}")

    if mlflow_run is not None:
        import mlflow

        final = {k: float(v[-1]) for k, v in history.history.items()}
        mlflow.log_metrics(final)
        mlflow.end_run()


if __name__ == "__main__":
    main()
