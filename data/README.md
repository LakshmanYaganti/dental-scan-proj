# Data

The dataset is **not bundled** with this repository (licensing + size + privacy).

## Expected layout

```
data/
  train/
    caries/
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

Images: `.jpg`, `.jpeg`, or `.png`. One class folder per condition.

## Where to get data

Use a publicly available dental panoramic X-ray dataset whose licence permits research use — for example Kaggle dental X-ray collections or Roboflow Universe dental exports. Check the licence and citation requirements on the dataset page before downloading.

## Privacy

- De-identify images before use: strip DICOM metadata and any burned-in identifiers.
- Never commit real patient data to this repository. `data/` is git-ignored (except this README).
- Keep raw downloads outside the repo where possible and symlink or copy only the de-identified split.

## Checking your split

```python
from dental_scan.data import count_images
print(count_images("data/train"))  # watch for severe class imbalance
```
