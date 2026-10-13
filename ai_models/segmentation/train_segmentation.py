
import csv
import json
import random
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import tensorflow as tf
from PIL import Image


# ---------------- Configuration ----------------

SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 20
PATIENCE = 5

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

SPLIT_CSV = SCRIPT_DIR / "splits" / "isic_segmentation_split.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "segmentation"

BEST_MODEL_PATH = OUTPUT_DIR / "isic_unet_split_best.keras"
FINAL_MODEL_PATH = OUTPUT_DIR / "isic_unet_split_final.keras"


# ---------------- Folder selection ----------------

def select_folder(title):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder = filedialog.askdirectory(title=title)

    root.destroy()

    if not folder:
        raise SystemExit("Folder selection cancelled.")

    return Path(folder)


# ---------------- Load and verify split ----------------

def load_pairs(image_dir, mask_dir):
    if not SPLIT_CSV.is_file():
        raise FileNotFoundError(
            "Split CSV missing. Run split_segmentation_dataset.py first."
        )

    pairs = {
        "train": [],
        "validation": [],
        "test": [],
    }

    with SPLIT_CSV.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {"split", "image", "mask"}

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Split CSV columns are incorrect.")

        for row in reader:
            split = row["split"]

            if split not in pairs:
                raise ValueError(f"Unknown split: {split}")

            image_path = image_dir / row["image"]
            mask_path = mask_dir / row["mask"]

            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image not found: {image_path.name}"
                )

            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Mask not found: {mask_path.name}"
                )

            expected_mask = f"{image_path.stem}_segmentation.png"

            if mask_path.name != expected_mask:
                raise ValueError(
                    f"Image-mask ID mismatch: {image_path.name}"
                )

            pairs[split].append((image_path, mask_path))

    if any(not pairs[name] for name in pairs):
        raise ValueError("One or more splits are empty.")

    split_ids = {
        name: {image.name for image, _ in entries}
        for name, entries in pairs.items()
    }

    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise ValueError("Data leakage detected between splits.")

    total = sum(len(entries) for entries in pairs.values())

    if total != len(set.union(*split_ids.values())):
        raise ValueError("Duplicate image IDs detected.")

    return pairs


# ---------------- Image preprocessing ----------------

def load_pair(image_path, mask_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )
        image = np.asarray(image, dtype=np.float32) / 255.0

    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        mask = mask.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST,
        )
        mask = np.asarray(mask, dtype=np.uint8)
        mask = (mask >= 128).astype(np.float32)
        mask = mask[..., np.newaxis]

    return image, mask


# ---------------- TensorFlow datasets ----------------

def make_dataset(pairs, training=False):
    pairs = list(pairs)

    def generator():
        for image_path, mask_path in pairs:
            image, mask = load_pair(image_path, mask_path)

            # Apply the same horizontal flip to image and mask.
            if training and random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            yield image, mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=(*IMAGE_SIZE, 3),
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(*IMAGE_SIZE, 1),
                dtype=tf.float32,
            ),
        ),
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(pairs), 512),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    # Repeat prevents the generator from exhausting between epochs.
    dataset = dataset.repeat()
    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )
    dataset = dataset.prefetch(1)

    return dataset


# ---------------- U-Net model ----------------

def build_unet():
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(
            filters, 3,
            padding="same",
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Conv2D(
            filters, 3,
            padding="same",
            activation="relu",
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)

        return x

    c1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    bridge = conv_block(p3, 128)

    u3 = tf.keras.layers.UpSampling2D()(bridge)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, 64)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 32)

    u1 = tf.keras.layers.UpSampling2D()(c5)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, 16)

    outputs = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid"
    )(c6)

    return tf.keras.Model(
        inputs,
        outputs,
        name="SkinSense_UNet",
    )


# ---------------- Loss and metric ----------------

def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred >= 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return (
        2.0 * intersection + 1e-6
    ) / (
        total + 1e-6
    )


def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        total + 1e-6
    )

    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )

    return 1.0 - dice + bce


# ---------------- Training ----------------

def main():
    print("Checking split CSV...")

    image_dir = select_folder(
        "Select ISIC Task 1 input images folder"
    )
    mask_dir = select_folder(
        "Select ISIC Task 1 ground-truth masks folder"
    )

    pairs = load_pairs(image_dir, mask_dir)

    print("\n--- Dataset Split ---")
    for name in ("train", "validation", "test"):
        print(f"{name.capitalize()}: {len(pairs[name])}")

    print("\nTest data will not be used for training or model selection.")

    train_ds = make_dataset(
        pairs["train"],
        training=True,
    )
    val_ds = make_dataset(
        pairs["validation"],
        training=False,
    )

    train_steps = int(
        np.ceil(len(pairs["train"]) / BATCH_SIZE)
    )
    val_steps = int(
        np.ceil(len(pairs["validation"]) / BATCH_SIZE)
    )

    print(f"Training steps per epoch: {train_steps}")
    print(f"Validation steps per epoch: {val_steps}")

    model = build_unet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss=dice_loss,
        metrics=[dice_coefficient],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_dice_coefficient",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(OUTPUT_DIR / "split_training_history.csv")
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(FINAL_MODEL_PATH)

    best_epoch = int(
        np.argmax(history.history["val_dice_coefficient"])
    )

    report = {
        "dataset": "ISIC 2018 Task 1",
        "train_images": len(pairs["train"]),
        "validation_images": len(pairs["validation"]),
        "test_images_reserved": len(pairs["test"]),
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": best_epoch + 1,
        "best_validation_dice": float(
            history.history["val_dice_coefficient"][best_epoch]
        ),
        "best_validation_loss": float(
            history.history["val_loss"][best_epoch]
        ),
        "test_used_for_training": False,
    }

    with (
        OUTPUT_DIR / "split_training_report.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n--- Training Completed ---")
    print(json.dumps(report, indent=4))
    print("\nBest model:", BEST_MODEL_PATH)
    print("Final model:", FINAL_MODEL_PATH)


if __name__ == "__main__":
    main()
