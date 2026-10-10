"""tf.data pipelines built from the folder layout:

    dataset/train/<class>/*.jpg
    dataset/validation/<class>/*.jpg
"""
import tensorflow as tf
from tensorflow import keras


def load_datasets(data_dir, img_size=224, batch_size=32, seed=42):
    train = keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val = keras.utils.image_dataset_from_directory(
        f"{data_dir}/validation",
        label_mode="categorical",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=False,
    )
    class_names = list(train.class_names)
    if class_names != list(val.class_names):
        raise ValueError("train/ and validation/ must contain the same class folders")
    autotune = tf.data.AUTOTUNE
    return train.prefetch(autotune), val.prefetch(autotune), class_names
