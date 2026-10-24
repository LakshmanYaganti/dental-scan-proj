"""FastAPI inference service.

Run::

    uvicorn dental_scan.api:app --app-dir src --host 0.0.0.0 --port 8000

Endpoints:
- ``GET  /health``  — always returns 200; reports whether a model is loaded.
- ``POST /predict`` — multipart image upload; returns predicted class +
  confidence. Returns 503 (not a crash) when no trained model exists.

Research / education only — not a medical device.
"""
from __future__ import annotations

import io
import json
import os
import pathlib

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_PATH = os.environ.get("DENTAL_MODEL_PATH", "models/dental_scan.keras")
LABELS_PATH = os.environ.get("DENTAL_LABELS_PATH", "models/class_labels.json")
IMAGE_SIZE = (224, 224)

app = FastAPI(
    title="Dental Scan API",
    description="Panoramic dental X-ray classification — research/education only, not a medical device.",
    version="0.1.0",
)

_model = None
_class_names: list[str] = ["caries", "healthy", "impacted_tooth", "implant"]


def _load_model():
    """Load the model lazily; return None (never raise) if unavailable."""
    global _model, _class_names
    if _model is not None:
        return _model
    if not pathlib.Path(MODEL_PATH).exists():
        return None
    try:
        import tensorflow as tf

        _model = tf.keras.models.load_model(MODEL_PATH)
        labels_file = pathlib.Path(LABELS_PATH)
        if labels_file.is_file():
            _class_names = json.loads(labels_file.read_text(encoding="utf-8"))
        return _model
    except Exception:
        return None


@app.on_event("startup")
def _startup() -> None:
    _load_model()  # best-effort; /health reports the outcome


@app.get("/health")
def health() -> dict:
    model = _load_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "classes": _class_names if model is not None else [],
        "disclaimer": "Research/education only — not a medical device.",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    model = _load_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model available. Train with "
                "`python -m dental_scan.train --config configs/config.yaml` "
                f"and ensure the model exists at {MODEL_PATH}."
            ),
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        import numpy as np
        import tensorflow as tf  # noqa: F401  (ensures TF errors surface as 400 below)

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        arr = np.expand_dims(np.asarray(image, dtype="float32"), axis=0)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    probs = model.predict(arr, verbose=0)[0]
    idx = int(probs.argmax())
    return {
        "predicted_class": _class_names[idx] if idx < len(_class_names) else str(idx),
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {
            (_class_names[i] if i < len(_class_names) else str(i)): round(float(p), 4)
            for i, p in enumerate(probs)
        },
        "disclaimer": "Research/education only — not a medical diagnosis.",
    }
