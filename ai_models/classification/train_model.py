
from pathlib import Path
import os
import random
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# 1. CONFIGURATION

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 5
FINE_TUNE_LAST_LAYERS = 30

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

ROOT = Path(__file__).resolve().parents[2]

SPLIT_CSV = ROOT / "data" / "splits" / "lesion_level_split.csv"
IMAGE_ROOT = ROOT / "Small HAM"

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "ai_models" / "evaluation" / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL = MODEL_DIR / "skinsense_best.keras"
FINAL_MODEL = MODEL_DIR / "skinsense_efficientnetb0.keras"

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

print("=" * 65)
print("SkinSense AI - Lesion-Level Training")
print("=" * 65)


# 2. LOAD SPLIT FILE

if not SPLIT_CSV.exists():
    raise FileNotFoundError(
        f"Split CSV not found:\n{SPLIT_CSV}"
    )

df = pd.read_csv(SPLIT_CSV)

print("\nCSV columns:", df.columns.tolist())


def find_column(candidates, required=True):
    columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in columns:
            return columns[candidate.lower()]

    if required:
        raise ValueError(
            f"Column not found: {candidates}\n"
            f"CSV columns: {df.columns.tolist()}"
        )

    return None


IMAGE_COL = find_column([
    "image_path", "filepath", "file_path",
    "path", "image", "filename", "image_id"
])

LABEL_COL = find_column([
    "folder_class", "class_name", "label", "dx", "class"
])

SPLIT_COL = find_column([
    "split", "set", "subset", "partition"
])

LESION_COL = find_column(
    ["lesion_id", "lesion"],
    required=False
)

df[LABEL_COL] = (
    df[LABEL_COL].astype(str).str.strip().str.upper()
)


def normalize_split(value):
    value = str(value).strip().lower()

    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }

    return aliases.get(value, value)


df[SPLIT_COL] = df[SPLIT_COL].apply(normalize_split)

df = df[df[LABEL_COL].isin(CLASS_NAMES)].copy()

if df.empty:
    raise ValueError("No valid class rows found in split CSV.")

if df[SPLIT_COL].isna().any():
    raise ValueError("Missing split values in CSV.")

if df.duplicated(subset=[IMAGE_COL]).any():
    raise ValueError("Duplicate image entries found in split CSV.")

required_splits = {"train", "validation", "test"}

if not required_splits.issubset(set(df[SPLIT_COL])):
    raise ValueError(
        "CSV must contain train, validation and test splits."
    )


# 3. RESOLVE IMAGE PATHS

def resolve_image_path(row):
    raw = str(row[IMAGE_COL]).strip()
    class_name = row[LABEL_COL]

    # Check path stored in CSV.
    path = Path(raw)

    if path.is_file():
        return str(path.resolve())

    # Check path relative to project root.
    path = ROOT / raw

    if path.is_file():
        return str(path.resolve())

    # Otherwise search in the appropriate class folder.
    filename = Path(raw.replace("\\", "/")).name
    class_dir = IMAGE_ROOT / class_name

    candidates = []

    if Path(filename).suffix:
        candidates.append(class_dir / filename)
    else:
        for extension in [".jpg", ".jpeg", ".png"]:
            candidates.append(class_dir / f"{filename}{extension}")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    raise FileNotFoundError(
        f"Image not found: {raw}\n"
        f"Class folder checked: {class_dir}"
    )


print("\nResolving image paths...")

df["resolved_path"] = df.apply(resolve_image_path, axis=1)

df["class_index"] = df[LABEL_COL].map(CLASS_TO_INDEX)

if df["resolved_path"].duplicated().any():
    raise ValueError("The same image path appears more than once.")

print("All image paths verified:", len(df))


# 4. VERIFY LESION-LEVEL SEPARATION

if LESION_COL is not None:
    if df[LESION_COL].isna().any():
        raise ValueError("Missing lesion IDs in split CSV.")

    df[LESION_COL] = df[LESION_COL].astype(str).str.strip()

    if df[LESION_COL].isin(["", "nan", "None"]).any():
        raise ValueError("Empty lesion IDs found.")

    train_lesions = set(
        df.loc[df[SPLIT_COL] == "train", LESION_COL]
    )
    val_lesions = set(
        df.loc[df[SPLIT_COL] == "validation", LESION_COL]
    )
    test_lesions = set(
        df.loc[df[SPLIT_COL] == "test", LESION_COL]
    )

    if train_lesions & val_lesions:
        raise ValueError("Lesion leakage between train and validation.")

    if train_lesions & test_lesions:
        raise ValueError("Lesion leakage between train and test.")

    if val_lesions & test_lesions:
        raise ValueError("Lesion leakage between validation and test.")

    print("PASS: No lesion crosses split boundaries.")

else:
    print(
        "WARNING: lesion_id column not found. "
        "Independent leakage verification is unavailable."
    )


train_df = df[df[SPLIT_COL] == "train"].copy()
val_df = df[df[SPLIT_COL] == "validation"].copy()
test_df = df[df[SPLIT_COL] == "test"].copy()

for name, part in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df),
]:
    counts = part[LABEL_COL].value_counts()

    missing = [
        class_name for class_name in CLASS_NAMES
        if counts.get(class_name, 0) == 0
    ]

    print(f"\n{name}: {len(part)} images")
    print(counts.reindex(CLASS_NAMES, fill_value=0).to_string())

    if part.empty or missing:
        raise ValueError(
            f"{name} has missing classes: {missing}"
        )


# 5. BUILD TENSORFLOW DATASETS

AUTOTUNE = tf.data.AUTOTUNE


def load_image(path, label):
    image_data = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_data,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True
    )

    # EfficientNetB0 includes its own input rescaling.
    image = tf.cast(image, tf.float32)

    return image, tf.cast(label, tf.int32)


def make_dataset(part, training=False):
    paths = part["resolved_path"].astype(str).to_numpy()
    labels = part["class_index"].astype("int32").to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            len(part),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)


# 6. CLASS IMBALANCE HANDLING

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=train_df["class_index"].to_numpy()
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(weights)
}

print("\nClass weights:")

for name, index in CLASS_TO_INDEX.items():
    print(f"{name}: {class_weights[index]:.3f}")


# 7. BUILD EFFICIENTNETB0

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
    ],
    name="data_augmentation"
)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMAGE_SIZE, 3)
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
    kernel_regularizer=tf.keras.regularizers.l2(0.001)
)(x)

x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="class_predictions"
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="SkinSense_EfficientNetB0"
)


def compile_model(learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


compile_model(1e-3)

model.summary()


# 8. CALLBACK FACTORY

def make_callbacks(initial_best=None, append_history=False):
    checkpoint_args = {
        "filepath": str(BEST_MODEL),
        "monitor": "val_loss",
        "save_best_only": True,
        "mode": "min",
        "verbose": 1,
    }

    # Preserve the best checkpoint from phase 1 during fine-tuning.
    if initial_best is not None:
        checkpoint_args["initial_value_threshold"] = float(initial_best)

    return [
        tf.keras.callbacks.ModelCheckpoint(**checkpoint_args),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),

        tf.keras.callbacks.CSVLogger(
            str(RESULT_DIR / "training_history.csv"),
            append=append_history
        ),
    ]


# 9. PHASE 1 - TRAIN CLASSIFICATION HEAD

print("\n" + "=" * 65)
print("PHASE 1: Training classification head")
print("=" * 65)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(append_history=False),
    verbose=1
)

best_head_val_loss = min(history_head.history["val_loss"])

print("\nBest phase 1 validation loss:", best_head_val_loss)


# 10. PHASE 2 - FINE-TUNE LAST LAYERS

print("\n" + "=" * 65)
print("PHASE 2: Fine-tuning EfficientNetB0")
print("=" * 65)

# Restore the best model obtained in phase 1.
model = tf.keras.models.load_model(BEST_MODEL)

base_model = model.get_layer("efficientnetb0")
base_model.trainable = True

# Freeze most layers; fine-tune only the last layers.
for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-FINE_TUNE_LAST_LAYERS:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

compile_model(1e-5)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=make_callbacks(
        initial_best=best_head_val_loss,
        append_history=True
    ),
    verbose=1
)


# 11. LOAD BEST CHECKPOINT

best_model = tf.keras.models.load_model(BEST_MODEL)
best_model.save(FINAL_MODEL)

print("\nBest model saved:", BEST_MODEL)
print("Final model saved:", FINAL_MODEL)


# 12. FINAL TEST EVALUATION

print("\n" + "=" * 65)
print("FINAL TEST EVALUATION")
print("=" * 65)

y_true = test_df["class_index"].to_numpy()

probabilities = best_model.predict(test_ds, verbose=1)
y_pred = np.argmax(probabilities, axis=1)

test_accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

balanced_acc = balanced_accuracy_score(y_true, y_pred)

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    zero_division=0
)

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

print(f"\nTest images: {len(test_df)}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
print(f"Macro F1-score: {macro_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

print("\nClass-wise report:")
print(report_text)


# 13. SAVE METRICS

metrics = {
    "model": "EfficientNetB0",
    "dataset": "Small HAM",
    "split_method": "lesion-level",
    "train_images": int(len(train_df)),
    "validation_images": int(len(val_df)),
    "test_images": int(len(test_df)),
    "test_accuracy": float(test_accuracy),
    "macro_f1": float(macro_f1),
    "balanced_accuracy": float(balanced_acc),
    "class_names": CLASS_NAMES,
    "classification_report": report_dict,
}

metrics_path = RESULT_DIR / "test_metrics.json"

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved:", metrics_path)


# 14. SAVE CONFUSION MATRIX

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASS_NAMES)))
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
    title="SkinSense AI - Lesion-Level Test Confusion Matrix"
)

plt.setp(
    ax.get_xticklabels(),
    rotation=45,
    ha="right",
    rotation_mode="anchor"
)

threshold = cm.max() / 2

for row in range(cm.shape[0]):
    for column in range(cm.shape[1]):
        ax.text(
            column,
            row,
            str(cm[row, column]),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[row, column] > threshold
                else "black"
            )
        )

fig.tight_layout()

cm_path = RESULT_DIR / "confusion_matrix.png"
fig.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Confusion matrix saved:", cm_path)

print("\n" + "=" * 65)
print("TRAINING AND EVALUATION COMPLETED")
print("=" * 65)
