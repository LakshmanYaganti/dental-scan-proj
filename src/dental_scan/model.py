"""Transfer-learning model builder.

Supports EfficientNetB0 (default) and MobileNetV2 backbones with
ImageNet weights, a frozen backbone, and a small classification head.
"""
from __future__ import annotations

try:
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "TensorFlow is required to build the model. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

_BACKBONES = {
    "efficientnetb0": tf.keras.applications.EfficientNetB0,
    "mobilenetv2": tf.keras.applications.MobileNetV2,
}


def build_model(
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    backbone: str = "efficientnetb0",
    dropout: float = 0.3,
    learning_rate: float = 1e-3,
    trainable_backbone: bool = False,
) -> tf.keras.Model:
    """Build a transfer-learning classification model.

    Parameters
    ----------
    num_classes:
        Number of output classes (one folder per class in the dataset).
    image_size:
        ``(height, width)`` of input images.
    backbone:
        ``"efficientnetb0"`` or ``"mobilenetv2"`` (case-insensitive).
    dropout:
        Dropout rate in the classification head.
    learning_rate:
        Adam learning rate.
    trainable_backbone:
        If False (phase 1), the backbone is frozen. Set True for
        phase-2 fine-tuning (use a lower learning rate).
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")

    key = backbone.lower().replace("-", "").replace("_", "")
    backbone_fn = _BACKBONES.get(key)
    if backbone_fn is None:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from: {sorted(_BACKBONES)}"
        )

    base = backbone_fn(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
    )
    base.trainable = trainable_backbone

    inputs = tf.keras.Input(shape=(*image_size, 3), name="image")
    # Backbones include their own preprocessing expectations; rescaling
    # to [0, 1] keeps the pipeline simple and reproducible.
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"dental_scan_{key}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: tf.keras.Model, n_layers: int = 20) -> tf.keras.Model:
    """Unfreeze the top ``n_layers`` of the backbone for fine-tuning.

    Call this after phase-1 training, then recompile with a low LR.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sub in layer.layers[-n_layers:]:
                sub.trainable = True
    return model
