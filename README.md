# Dental Scan — Panoramic X-Ray Image Classification

> Research / education project for classifying dental panoramic X-ray images into common conditions using transfer learning. **Not a medical device. Not for clinical diagnosis.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Problem

Dental panoramic X-rays (orthopantomograms) are a first-line screening view, but reading them is time-consuming and varies by clinician experience. This project explores whether a transfer-learning CNN can provide a reproducible, auditable baseline for classifying cropped / full panoramic images into a small set of conditions:

| Class | Description |
|---|---|
| `caries` | Visible dental caries / decay |
| `impacted_tooth` | Impacted tooth (e.g., third molar) |
| `implant` | Dental implant present |
| `healthy` | No finding in the above categories |

The goal is **decision-support research and education** — triage prioritisation, dataset curation, and model-evaluation practice — not automated diagnosis.

## Approach

```
Public dental X-ray dataset (not bundled)
        │
        ▼
data/  (train / val / test, one folder per class)
        │
        ▼
tf.keras image_dataset_from_directory  +  augmentation
(RandomFlip, RandomRotation, RandomZoom, RandomContrast)
        │
        ▼
Transfer learning backbone (EfficientNetB0 or MobileNetV2, ImageNet weights)
  Phase 1: frozen backbone, train classification head
  Phase 2 (optional): fine-tune top layers at low LR
        │
        ▼
Evaluation: accuracy, per-class precision/recall/F1, confusion matrix
        │
        ▼
Inference: predict.py (single image)  +  api.py (FastAPI /predict)
        │
        ▼
Experiment tracking: MLflow (optional, guarded — training works without it)
```

Key design choices:
- **Small, inspectable codebase** — data, model, train, evaluate, predict, and API are separate modules under `src/dental_scan/`.
- **Config-driven** — image size, backbone, epochs, batch size, and paths live in `configs/config.yaml`.
- **Honest evaluation** — metrics are computed on a held-out split; no test-set tuning.
- **Graceful degradation** — the API starts even when no trained model exists and returns a clear `503` on `/predict` until a model is trained.

## Dataset

**The dataset is NOT bundled with this repository** for licensing and size reasons.

### Where to obtain a public dataset

Use a publicly available dental X-ray dataset whose licence permits research use, for example:

- Kaggle — *Dental Xray Images* / *Panoramic Dental X-ray* collections (check each dataset's licence on its Kaggle page before use)
- Roboflow Universe — dental / panoramic X-ray classification exports
- Institutional open datasets released for research (e.g., the UF / Tufts dental datasets where available)

Always confirm the licence, citation requirements, and whether redistribution is allowed. Do not commit patient-identifiable data to this repo.

### Expected folder layout

```
data/
  train/
    caries/           *.jpg / *.png
    healthy/
    impacted_tooth/
    implant/
  val/
    caries/
    healthy/
    impacted_tooth/
    implant/
  test/
    caries/
    healthy/
    impacted_tooth/
    implant/
```

If your source dataset is a single folder-per-class tree, split it 70 / 15 / 15 (train / val / test) stratified by class, or point `train.py` at the root and let the loader create a validation split (`validation_split` in config).

De-identification note: strip DICOM metadata and any burned-in identifiers before training, and keep raw data outside version control (`data/` is git-ignored except this README).

## Project Structure

```
dental-scan-proj/
├── configs/
│   └── config.yaml          # Training / model / data configuration
├── data/
│   └── README.md            # Dataset instructions (data itself is git-ignored)
├── notebooks/
│   └── README.md            # Planned EDA notebooks
├── src/
│   └── dental_scan/
│       ├── __init__.py
│       ├── data.py          # Dataset loading + augmentation
│       ├── model.py         # Transfer-learning model builder
│       ├── train.py         # CLI training entrypoint (optional MLflow)
│       ├── evaluate.py      # Classification report + confusion matrix
│       ├── predict.py       # Single-image inference CLI
│       └── api.py           # FastAPI inference service
├── tests/
│   └── test_smoke.py        # Smoke tests (config, data utils, model builder)
├── requirements.txt
├── LICENSE                  # MIT
└── README.md
```

## Setup

```bash
git clone https://github.com/LakshmanYaganti/dental-scan-proj.git
cd dental-scan-proj

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python -m dental_scan.train --config configs/config.yaml
# or, from src layout:
PYTHONPATH=src python -m dental_scan.train --config configs/config.yaml
```

Outputs:
- `models/dental_scan.keras` — trained model
- `models/class_labels.json` — ordered class labels
- MLflow run (only if `mlflow` is installed and tracking is enabled in config)

## Evaluation

```bash
PYTHONPATH=src python -m dental_scan.evaluate \
  --config configs/config.yaml \
  --model models/dental_scan.keras \
  --data-dir data/test
```

Prints a scikit-learn classification report and writes `reports/confusion_matrix.png` and `reports/classification_report.txt` when an output directory is configured.

## Inference

### Single image (CLI)

```bash
PYTHONPATH=src python -m dental_scan.predict \
  --model models/dental_scan.keras \
  --image path/to/xray.jpg
# {"predicted_class": "caries", "confidence": 0.87, "probabilities": {...}}
```

### HTTP API (FastAPI)

```bash
uvicorn dental_scan.api:app --app-dir src --host 0.0.0.0 --port 8000
```

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}

curl -X POST http://localhost:8000/predict -F "file=@xray.jpg"
# {"predicted_class": "caries", "confidence": 0.87, "probabilities": {...}}
```

If no trained model is present, `GET /health` still returns `200` with `"model_loaded": false`, and `POST /predict` returns `503` with an explanation — the service never crashes on startup for a missing model.

Interactive docs: http://localhost:8000/docs

## Results

> **No performance is claimed for this repository yet.** The training and evaluation pipeline is complete, but it has not been run on a licensed dental X-ray dataset, so there are no measured results to report — and none are invented here.

**Evaluation protocol** — run `src/train.py`, then `src/evaluate.py`, on your licensed split:

- Report accuracy **and** macro-F1, per-class precision/recall, and the confusion matrix. A single accuracy number on an imbalanced dental dataset is misleading.
- Compare the three shipped configurations: MobileNetV2 (frozen), EfficientNetB0 (frozen), and EfficientNetB0 (fine-tuned, low LR).
- Publish the dataset name, split sizes, and class balance next to the metrics, so the numbers can be reproduced.

## Ethics & Medical Disclaimer

- This is a **research and education project**. It is **not a medical device**, has not been clinically validated, and must not be used to diagnose, treat, or triage real patients.
- Dental X-ray datasets can encode bias (age, anatomy, scanner, clinic, geography). Evaluate per-group performance before drawing any conclusions.
- Respect dataset licences, patient privacy, and de-identification requirements. Never upload identifiable patient images to this repository or to third-party services without a lawful basis.
- Any future clinical direction would require regulatory review, prospective validation, and clinician oversight — all out of scope here.

## Author

**Lakshman Sai Yaganti** — Data Scientist (FinTech, payments, fraud & credit risk; ML / NLP)
- GitHub: https://github.com/LakshmanYaganti
- LinkedIn: https://www.linkedin.com/in/y-lakshman/
- Portfolio: https://lakshmanyagantiportfolio.lovable.app/
- Email: lakshmanyaganti47@gmail.com

## License

MIT — see [LICENSE](LICENSE).
