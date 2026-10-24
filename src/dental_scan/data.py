"""Dataset loading and augmentation utilities.

Expects a folder-per-class layout, e.g.::

    data/train/caries/*.jpg
    data/train/healthy/*.jpg

Uses ``tf.keras.utils.image_dataset_from_directory`` so no custom
image decoding code is needed.
"""
from __future__ import annotations

import pathlib
from typing import Any

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required for data loading. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")


def list_classes(data_dir: str | pathlib.Path) -> list[str]:
    """Return sorted class names from a folder-per-class directory."""
    root = pathlib.Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class sub-folders found in {root}")
    return classes


def count_images(data_dir: str | pathlib.Path) -> dict[str, int]:
    """Count images per class — useful for spotting class imbalance."""
    root = pathlib.Path(data_dir)
    counts: dict[str, int] = {}
    for cls in list_classes(root):
        files = [
            f
            for f in (root / cls).rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        counts[cls] = len(files)
    return counts


def build_augmentation() -> tf.keras.Sequential:
    """Light, label-preserving augmentation for X-ray images.

    Horizontal flips are plausible for panoramic views; large rotations
    are not, so rotation / zoom ranges are deliberately small.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )


def load_dataset(
    data_dir: str | pathlib.Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
    validation_split: float | None = None,
    subset: str | None = None,
    augment: bool = False,
) -> Any:
    """Load a folder-per-class dataset as a ``tf.data.Dataset``.

    Returns ``(dataset, class_names)``.
    """
    kwargs: dict[str, Any] = dict(
        directory=str(data_dir),
        labels="inferred",
        label_mode="int",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    if validation_split:
        kwargs["validation_split"] = validation_split
        kwargs["subset"] = subset

    dataset = tf.keras.utils.image_dataset_from_directory(**kwargs)
    class_names = list(dataset.class_names)

    if augment:
        aug = build_augmentation()
        dataset = dataset.map(
            lambda x, y: (aug(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset, class_names


def load_train_val(
    data_dir: str | pathlib.Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.15,
    seed: int = 42,
) -> tuple[Any, Any, list[str]]:
    """Convenience helper: train (augmented) + validation split."""
    train_ds, class_names = load_dataset(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
        subset="training",
        augment=True,
    )
    val_ds, _ = load_dataset(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
        shuffle=False,
        augment=False,
    )
    return train_ds, val_ds, class_names
