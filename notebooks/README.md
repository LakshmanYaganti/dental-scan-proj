# Notebooks

Planned exploratory notebooks (kept out of `src/` so the package stays importable and testable):

| Notebook | Purpose |
|---|---|
| `01_eda_dataset_overview.ipynb` | Class balance, image sizes / aspect ratios, sample grids per class, duplicate & corrupt-image checks |
| `02_visualise_augmentation.ipynb` | Preview augmentation outputs to confirm transforms stay label-preserving for panoramic X-rays |
| `03_error_analysis.ipynb` | Confusion-matrix deep dive, misclassified examples, confidence calibration, per-class failure patterns |
| `04_gradcam_explainability.ipynb` | Grad-CAM overlays to sanity-check that the model attends to tooth / jaw regions rather than image artefacts |

## Conventions

- Notebooks read paths from `../configs/config.yaml` — no hard-coded absolute paths.
- Notebooks are for exploration only. Anything that matters for reproducibility belongs in `src/dental_scan/` with a test.
- Do not embed real patient images in committed notebook outputs; clear outputs before committing if samples are sensitive.
