import csv
import random
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# CONFIGURATION

METADATA_PATH = Path(
    r"C:\Users\HELLO\OneDrive\Desktop\Skin Diseases\HAM10000\HAM10000_metadata.csv"
)

# Common parent of HAM10000_images_part_1 and part_2
IMAGE_ROOT = Path(
    r"C:\Users\HELLO\OneDrive\Desktop\Skin Diseases\HAM10000"
)

OUTPUT_PARENT = Path(r"F:\SkinSense-Datasets\Balance Dataset")

CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

IMAGE_SIZE = (320, 320)
TARGET_TRAIN_IMAGES_PER_CLASS = 1000
SEED = 42

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

random.seed(SEED)
np.random.seed(SEED)


# INDEX ORIGINAL IMAGES

def index_images(root_folder):
    image_index = defaultdict(list)

    print("\nScanning image folders recursively...")

    for path in root_folder.rglob("*"):

        # Ignore previously generated balanced datasets.
        if any(
            part.startswith("HAM10000_Balanced_")
            for part in path.parts
        ):
            continue

        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            image_index[path.stem].append(path)

    for image_id in image_index:
        image_index[image_id] = sorted(
            image_index[image_id],
            key=lambda p: str(p).lower()
        )

    print(f"Unique image IDs found: {len(image_index)}")

    return image_index


# IMAGE LOADING AND SAVING

def load_image(image_path):
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image = ImageOps.fit(
            image,
            IMAGE_SIZE,
            method=Image.Resampling.LANCZOS
        )

    return image


def save_image(image, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)

    image.save(
        destination,
        format="JPEG",
        quality=95,
        optimize=True
    )


# STRONG BUT CONTROLLED AUGMENTATION

def augment_image(original):
    image = original.copy()
    width, height = image.size

    # Horizontal flip
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    # Vertical flip
    if random.random() < 0.25:
        image = ImageOps.flip(image)

    # Rotation
    angle = random.uniform(-180, 180)

    image = image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0)
    )

    # Random zoom-in crop
    if random.random() < 0.8:

        ratio = random.uniform(0.75, 0.97)

        crop_width = max(1, int(width * ratio))
        crop_height = max(1, int(height * ratio))

        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)

        image = image.crop((
            left,
            top,
            left + crop_width,
            top + crop_height
        ))

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

    # Brightness
    image = ImageEnhance.Brightness(image).enhance(
        random.uniform(0.85, 1.15)
    )

    # Contrast
    image = ImageEnhance.Contrast(image).enhance(
        random.uniform(0.85, 1.15)
    )

    # Colour variation
    image = ImageEnhance.Color(image).enhance(
        random.uniform(0.90, 1.10)
    )

    # Gamma variation
    if random.random() < 0.4:

        array = np.asarray(image).astype(np.float32) / 255.0
        gamma = random.uniform(0.85, 1.15)

        array = np.power(np.clip(array, 0, 1), gamma)

        image = Image.fromarray(
            (np.clip(array, 0, 1) * 255).astype(np.uint8)
        )

    # Mild blur
    if random.random() < 0.10:

        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(0.2, 0.6)
            )
        )

    # Low-level noise
    if random.random() < 0.15:

        array = np.asarray(image).astype(np.float32)

        noise = np.random.normal(
            0,
            random.uniform(1.0, 3.0),
            array.shape
        )

        image = Image.fromarray(
            np.clip(array + noise, 0, 255).astype(np.uint8)
        )

    return image


# LESION-LEVEL SPLIT

def create_lesion_level_split(df):

    # Every lesion must belong to only one split.
    lesion_diagnoses = (
        df.groupby("lesion_id")["dx"]
        .agg(lambda values: Counter(values).most_common(1)[0][0])
        .to_dict()
    )

    lesions_by_class = defaultdict(list)

    for lesion_id, diagnosis in lesion_diagnoses.items():
        lesions_by_class[diagnosis].append(lesion_id)

    lesion_split = {}

    for diagnosis in CLASSES:

        lesion_ids = sorted(lesions_by_class.get(diagnosis, []))
        random.shuffle(lesion_ids)

        total = len(lesion_ids)

        if total < 3:
            raise ValueError(
                f"Not enough unique lesions for {diagnosis}"
            )

        train_count = max(1, int(total * 0.70))
        validation_count = max(1, int(total * 0.15))

        if train_count + validation_count >= total:
            train_count = total - 2
            validation_count = 1

        for lesion_id in lesion_ids[:train_count]:
            lesion_split[lesion_id] = "train"

        for lesion_id in lesion_ids[
            train_count:train_count + validation_count
        ]:
            lesion_split[lesion_id] = "validation"

        for lesion_id in lesion_ids[
            train_count + validation_count:
        ]:
            lesion_split[lesion_id] = "test"

    df = df.copy()

    df["split"] = df["lesion_id"].map(lesion_split)

    if df["split"].isna().any():
        raise RuntimeError("Some lesions were not assigned to a split.")

    if (df.groupby("lesion_id")["split"].nunique() > 1).any():
        raise RuntimeError("Lesion leakage detected.")

    return df


# MAIN DATASET PREPARATION

def prepare_dataset():

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found:\n{METADATA_PATH}"
        )

    if not IMAGE_ROOT.is_dir():
        raise FileNotFoundError(
            f"HAM10000 folder not found:\n{IMAGE_ROOT}"
        )

    OUTPUT_PARENT.mkdir(parents=True, exist_ok=True)

    print("\nReading HAM10000 metadata...")

    df = pd.read_csv(METADATA_PATH)

    required_columns = {
        "image_id",
        "dx",
        "lesion_id"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing metadata columns: "
            + ", ".join(sorted(missing_columns))
        )

    df = df[["image_id", "dx", "lesion_id"]].copy()

    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["dx"] = df["dx"].astype(str).str.strip().str.upper()
    df["lesion_id"] = df["lesion_id"].astype(str).str.strip()

    if df[["image_id", "dx", "lesion_id"]].isna().any().any():
        raise ValueError("Missing required metadata values.")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs found in metadata.")

    df = df[df["dx"].isin(CLASSES)].copy()

    print(f"Metadata rows: {len(df)}")

    print("\nOriginal diagnosis counts:")

    print(
        df["dx"].value_counts()
        .reindex(CLASSES, fill_value=0)
    )

    # Scan both original image folders.
    image_index = index_images(IMAGE_ROOT)

    missing_ids = [
        image_id
        for image_id in df["image_id"]
        if image_id not in image_index
    ]

    # Never silently create an incomplete dataset.
    if missing_ids:

        print(f"\nMissing images: {len(missing_ids)}")

        print("Examples:", missing_ids[:10])

        raise FileNotFoundError(
            "\nSTOP: Not all metadata images were found. "
            "Check both HAM10000 image folders and ensure their files "
            "are locally available. No new dataset was created."
        )

    print("\nAll metadata images found.")

    df["source_path"] = df["image_id"].map(
        lambda image_id: str(image_index[image_id][0])
    )

    # Split original images before augmentation.
    df = create_lesion_level_split(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_root = (
        OUTPUT_PARENT / f"HAM10000_Balanced_{timestamp}"
    )

    output_root.mkdir(parents=True, exist_ok=False)

    # Create folders.
    for split in ["train", "validation", "test"]:

        for diagnosis in CLASSES:

            (
                output_root / split / diagnosis
            ).mkdir(parents=True, exist_ok=True)

    # Save split metadata without local absolute paths.
    df.drop(columns=["source_path"]).to_csv(
        output_root / "original_lesion_split.csv",
        index=False
    )

    manifest_rows = []

    # VALIDATION AND TEST: ORIGINAL IMAGES ONLY

    print("\nCopying original validation and test images...")

    evaluation_df = df[
        df["split"].isin(["validation", "test"])
    ]

    for row in evaluation_df.itertuples(index=False):

        image = load_image(Path(row.source_path))

        destination = (
            output_root / row.split / row.dx / f"{row.image_id}.jpg"
        )

        save_image(image, destination)

        manifest_rows.append({
            "filename": str(destination.relative_to(output_root)),
            "image_id": row.image_id,
            "lesion_id": row.lesion_id,
            "dx": row.dx,
            "split": row.split,
            "is_augmented": False,
            "source_image_id": row.image_id
        })

    # TRAIN: EXACTLY 1000 IMAGES PER CLASS

    print("\nCreating balanced training data...")

    train_df = df[df["split"] == "train"]

    for diagnosis in CLASSES:

        source_rows = train_df[
            train_df["dx"] == diagnosis
        ].to_dict("records")

        if not source_rows:
            raise ValueError(
                f"No training images available for {diagnosis}"
            )

        random.shuffle(source_rows)

        target = TARGET_TRAIN_IMAGES_PER_CLASS

        # Downsample classes with more than 1000 originals.
        if len(source_rows) > target:

            selected_rows = random.sample(
                source_rows,
                target
            )

        else:
            selected_rows = source_rows

        original_cache = {}
        saved_count = 0

        # Save originals first.
        for row in selected_rows:

            image_id = row["image_id"]

            image = load_image(
                Path(row["source_path"])
            )

            original_cache[image_id] = image

            destination = (
                output_root
                / "train"
                / diagnosis
                / f"original_{image_id}.jpg"
            )

            save_image(image, destination)

            manifest_rows.append({
                "filename": str(destination.relative_to(output_root)),
                "image_id": image_id,
                "lesion_id": row["lesion_id"],
                "dx": diagnosis,
                "split": "train",
                "is_augmented": False,
                "source_image_id": image_id
            })

            saved_count += 1

        # Keep generating variants until exactly 1000 files exist.
        aug_count = 0

        while saved_count < target:

            row = random.choice(selected_rows)

            source_id = row["image_id"]

            if source_id not in original_cache:

                original_cache[source_id] = load_image(
                    Path(row["source_path"])
                )

            augmented = augment_image(
                original_cache[source_id]
            )

            aug_count += 1

            destination = (
                output_root
                / "train"
                / diagnosis
                / f"aug_{aug_count:05d}_{source_id}.jpg"
            )

            save_image(augmented, destination)

            manifest_rows.append({
                "filename": str(destination.relative_to(output_root)),
                "image_id": "",
                "lesion_id": row["lesion_id"],
                "dx": diagnosis,
                "split": "train",
                "is_augmented": True,
                "source_image_id": source_id
            })

            saved_count += 1

        print(
            f"{diagnosis}: {saved_count} training images "
            f"({len(selected_rows)} original + "
            f"{aug_count} augmented)"
        )

    # SAVE MANIFEST

    manifest_path = output_root / "dataset_manifest.csv"

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "image_id",
                "lesion_id",
                "dx",
                "split",
                "is_augmented",
                "source_image_id"
            ]
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    # REPORT SPLIT COUNTS

    print("\nOriginal-image split counts:")

    print(
        df.groupby(["split", "dx"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASSES, fill_value=0)
    )

    # Verify actual files on disk.
    print("\nFinal training folder counts:")

    for diagnosis in CLASSES:

        folder = output_root / "train" / diagnosis

        count = sum(
            1
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
        )

        print(f"{diagnosis}: {count}")

        if count != TARGET_TRAIN_IMAGES_PER_CLASS:
            raise RuntimeError(
                f"{diagnosis} has {count} images, not {target}."
            )

    # Final lesion leakage check.
    if (df.groupby("lesion_id")["split"].nunique() > 1).any():
        raise RuntimeError("Lesion leakage detected.")

    print("\nLeakage check: OK")

    print("\nDataset preparation completed successfully.")

    print(f"Output folder: {output_root}")

    print(f"Image size: {IMAGE_SIZE}")

    print("Training: exactly 1000 images per class.")

    print("Validation/test: original images only.")

    print("Original HAM10000 files were not modified.")


# RUN

if __name__ == "__main__":

    try:
        prepare_dataset()

    except Exception as error:
        print(f"\nERROR: {error}")
        raise