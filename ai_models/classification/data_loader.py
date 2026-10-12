
from pathlib import Path
import tensorflow as tf

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_DIR / "Small HAM"

CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        validation_split=0.25,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels="inferred",
        label_mode="int",
        class_names=CLASSES,
        validation_split=0.25,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("\nClass names:", train_ds.class_names)
    print("Training batches:", len(train_ds))
    print("Validation batches:", len(val_ds))

    # Improve input pipeline performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = load_datasets()

    images, labels = next(iter(train_ds))

    print("\nOne batch image shape:", images.shape)
    print("One batch label shape:", labels.shape)
    print("Labels in first batch:", labels.numpy())
