"""Model definitions.

Two architectures are available:

* ``mobilenetv2`` (default): transfer learning from ImageNet. This is what fixes the
  heavy over-fitting seen with the original from-scratch CNN (train 99% vs val 65%).
* ``cnn``: a small from-scratch CNN (BatchNorm + GlobalAveragePooling + Dropout),
  kept for comparison / teaching.

Pre-processing (rescaling) and data augmentation live *inside* the model, so at
inference time you can feed raw RGB pixels in the 0-255 range.
"""
from tensorflow import keras
from tensorflow.keras import layers


def _augmentation():
    # Only active during training; automatically a no-op at inference.
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.1, 0.1),
            layers.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def build_model(num_classes, arch="mobilenetv2", img_size=224, pretrained=True, dropout=0.3):
    """Return ``(model, backbone)``. ``backbone`` is None for the plain CNN."""
    inputs = keras.Input(shape=(img_size, img_size, 3), name="image")
    x = _augmentation()(inputs)
    backbone = None

    if arch == "mobilenetv2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="rescale")(x)  # -> [-1, 1]
        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet" if pretrained else None,
            input_shape=(img_size, img_size, 3),
        )
        backbone.trainable = False  # phase 1: train only the new head
        x = backbone(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(dropout)(x)
    elif arch == "cnn":
        x = layers.Rescaling(1.0 / 255, name="rescale")(x)
        for filters in (32, 64, 128, 128):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(max(dropout, 0.5))(x)
    else:
        raise ValueError(f"Unknown arch '{arch}' (use 'mobilenetv2' or 'cnn')")

    outputs = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = keras.Model(inputs, outputs, name=f"food_{arch}")
    return model, backbone


def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top(backbone, n_layers=40):
    """Phase 2 (fine-tuning): unfreeze the last `n_layers` of the backbone.

    BatchNorm layers stay frozen, which is important for stable fine-tuning.
    """
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
