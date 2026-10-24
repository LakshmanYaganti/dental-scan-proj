"""Single-image inference.

Usage::

    PYTHONPATH=src python -m dental_scan.predict \
        --model models/dental_scan.keras \
        --image path/to/xray.jpg \
        --labels models/class_labels.json
"""
from __future__ import annotations

import argparse
import json
import pathlib


def load_labels(labels_path: str | pathlib.Path) -> list[str]:
    path = pathlib.Path(labels_path)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    # Fallback matching the README class set / alphabetical folder order.
    return ["caries", "healthy", "impacted_tooth", "implant"]


def predict_image(model_path: str, image_path: str, labels_path: str = "models/class_labels.json",
                  image_size: tuple[int, int] = (224, 224)) -> dict:
    """Return ``{predicted_class, confidence, probabilities}`` for one image."""
    import tensorflow as tf

    if not pathlib.Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train first with dental_scan.train."
        )
    if not pathlib.Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    class_names = load_labels(labels_path)
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.utils.load_img(image_path, target_size=image_size)
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.expand_dims(arr, 0)  # batch dim; model rescales internally

    probs = model.predict(arr, verbose=0)[0]
    idx = int(probs.argmax())
    return {
        "predicted_class": class_names[idx] if idx < len(class_names) else str(idx),
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {
            (class_names[i] if i < len(class_names) else str(i)): round(float(p), 4)
            for i, p in enumerate(probs)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single dental X-ray image.")
    parser.add_argument("--model", default="models/dental_scan.keras")
    parser.add_argument("--image", required=True)
    parser.add_argument("--labels", default="models/class_labels.json")
    args = parser.parse_args()
    result = predict_image(args.model, args.image, args.labels)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
