
from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 4

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LAST_LAYERS = 30

CLASS_NAMES = [
    "AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"
]

ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = Path(
    r"F:\SkinSense-Datasets\Balance Dataset"
    r"\HAM10000_Balanced_20261014_000220"
)

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "validation"
TEST_DIR = DATASET_ROOT / "test"

MODEL_DIR = ROOT / "models"
RESULT_DIR = (
    ROOT / "ai_models" / "evaluation" / "balanced_b3_results"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# B3 uses separate filenames; existing models remain untouched.
HEAD_MODEL = MODEL_DIR / "skinsense_b3_head_best.keras"
FINE_MODEL = MODEL_DIR / "skinsense_b3_finetune_best.keras"
BEST_MODEL = MODEL_DIR / "skinsense_efficientnetb3_balanced_best.keras"
FINAL_MODEL = MODEL_DIR / "skinsense_efficientnetb3_balanced_final.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 65)
print("SkinSense AI - EfficientNetB3 Balanced HAM10000")
print("=" * 65)

print("TensorFlow:", tf.__version__)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)

# ============================================================
# 2. VERIFY DATASET
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{folder}"
        )


def count_images(folder):
    return sum(
        1
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


for split_name, split_dir in [
    ("TRAIN", TRAIN_DIR),
    ("VALIDATION", VAL_DIR),
    ("TEST", TEST_DIR),
]:
    print(f"\n{split_name} dataset:")

    existing_classes = {
        folder.name
        for folder in split_dir.iterdir()
        if folder.is_dir()
    }

    missing = set(CLASS_NAMES) - existing_classes

    if missing:
        raise ValueError(
            f"{split_name} is missing classes: {sorted(missing)}"
        )

    for class_name in CLASS_NAMES:
        count = count_images(split_dir / class_name)
        print(f"  {class_name}: {count}")

        if count == 0:
            raise ValueError(
                f"No images found: {split_dir / class_name}"
            )

for class_name in CLASS_NAMES:
    count = count_images(TRAIN_DIR / class_name)

    if count != 1000:
        raise ValueError(
            f"{class_name}: expected 1000 training images, "
            f"but found {count}."
        )

print("\nDataset verification passed.")
print("Training images: 7000")

# ============================================================
# 3. LOAD DATASETS
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(VAL_DIR),
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    labels="inferred",
    label_mode="int",
    class_names=CLASS_NAMES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Conservative prefetch helps limit extra RAM use.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(1)
val_ds = val_ds.prefetch(1)
test_ds = test_ds.prefetch(1)

# ============================================================
# 4. BUILD EFFICIENTNETB3
# ============================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.08),
    ],
    name="b3_data_augmentation",
)

base_model = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3),
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.35)(x)

x = tf.keras.layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(0.001),
)(x)

x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="class_predictions",
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="SkinSense_EfficientNetB3_Balanced",
)


def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


compile_model(1e-3)
model.summary()

# ============================================================
# 5. PHASE 1 - CLASSIFICATION HEAD
# ============================================================

print("\n" + "=" * 65)
print("PHASE 1: Training classification head")
print("=" * 65)

head_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(HEAD_MODEL),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(RESULT_DIR / "b3_head_history.csv")
    ),
]

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=head_callbacks,
    verbose=1,
)

head_best_loss = float(min(history_head.history["val_loss"]))

print("\nBest head validation loss:", head_best_loss)

# ============================================================
# 6. PHASE 2 - FINE-TUNING
# ============================================================

print("\n" + "=" * 65)
print("PHASE 2: Fine-tuning EfficientNetB3")
print("=" * 65)

model = tf.keras.models.load_model(HEAD_MODEL)

base_model = model.get_layer("efficientnetb3")
base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-FINE_TUNE_LAST_LAYERS:]:
    if not isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

compile_model(1e-5)

fine_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(FINE_MODEL),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(RESULT_DIR / "b3_finetune_history.csv")
    ),
]

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=fine_callbacks,
    verbose=1,
)

fine_best_loss = (
    float(min(history_fine.history["val_loss"]))
    if history_fine.history["val_loss"]
    else float("inf")
)

# ============================================================
# 7. SELECT BEST VALIDATION CHECKPOINT
# ============================================================

print("\nHead best validation loss:", head_best_loss)
print("Fine-tuning best validation loss:", fine_best_loss)

if fine_best_loss < head_best_loss and FINE_MODEL.exists():
    best_model = tf.keras.models.load_model(FINE_MODEL)
    selected_phase = "fine_tuning"
else:
    best_model = tf.keras.models.load_model(HEAD_MODEL)
    selected_phase = "classification_head"

best_model.save(BEST_MODEL)
best_model.save(FINAL_MODEL)

print("\nSelected phase:", selected_phase)
print("Best model:", BEST_MODEL)
print("Final model:", FINAL_MODEL)

# ============================================================
# 8. FINAL TEST EVALUATION
# ============================================================

# Test data is not used for fitting or checkpoint selection.
print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds],
    axis=0,
)

probabilities = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

test_accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

balanced_acc = balanced_accuracy_score(y_true, y_pred)

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0,
)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

print(f"\nTest images: {len(y_true)}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

print("\nClass-wise report:")
print(report_text)

# ============================================================
# 9. SAVE METRICS
# ============================================================

metrics = {
    "model": "EfficientNetB3",
    "dataset": DATASET_ROOT.name,
    "selected_phase": selected_phase,
    "train_images": sum(
        count_images(TRAIN_DIR / c) for c in CLASS_NAMES
    ),
    "validation_images": sum(
        count_images(VAL_DIR / c) for c in CLASS_NAMES
    ),
    "test_images": int(len(y_true)),
    "test_accuracy": float(test_accuracy),
    "macro_f1": float(macro_f1),
    "balanced_accuracy": float(balanced_acc),
    "class_names": CLASS_NAMES,
    "classification_report": report_dict,
}

metrics_path = RESULT_DIR / "b3_test_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved:", metrics_path)

# ============================================================
# 10. SAVE CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
)

fig, ax = plt.subplots(figsize=(9, 7))

image = ax.imshow(cm, interpolation="nearest")
fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(len(CLASS_NAMES)),
    yticks=np.arange(len(CLASS_NAMES)),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted label",
    ylabel="True label",
    title="SkinSense AI - EfficientNetB3 Test Confusion Matrix",
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor",
)

threshold = cm.max() / 2 if cm.size else 0

for row in range(cm.shape[0]):
    for column in range(cm.shape[1]):
        ax.text(
            column,
            row,
            str(cm[row, column]),
            ha="center",
            va="center",
            color=(
                "white" if cm[row, column] > threshold
                else "black"
            ),
        )

fig.tight_layout()

cm_path = RESULT_DIR / "b3_confusion_matrix.png"
fig.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Confusion matrix saved:", cm_path)

print("\n" + "=" * 65)
print("EFFICIENTNETB3 TRAINING COMPLETED")
print("=" * 65)
