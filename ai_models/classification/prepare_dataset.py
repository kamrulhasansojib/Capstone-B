from pathlib import Path
import pandas as pd
import shutil

# Full HAM10000 dataset location
SOURCE_DIR = Path(
    r"C:\Users\HELLO\OneDrive\Desktop\Skin Diseases\HAM10000"
)

CSV_PATH = SOURCE_DIR / "HAM10000_metadata.csv"
OUTPUT_DIR = SOURCE_DIR / "HAM10000_Balanced"

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
IMAGES_PER_CLASS = 300


def main():
    # Check dataset and metadata
    if not SOURCE_DIR.exists():
        print("ERROR: Dataset folder not found:")
        print(SOURCE_DIR)
        return

    if not CSV_PATH.exists():
        print("ERROR: Metadata CSV not found:")
        print(CSV_PATH)
        print("Check the exact CSV filename.")
        return

    print("Reading HAM10000 metadata...")
    df = pd.read_csv(CSV_PATH)

    required_columns = {"image_id", "dx"}
    if not required_columns.issubset(df.columns):
        print("ERROR: CSV must contain image_id and dx columns.")
        print("Available columns:", list(df.columns))
        return

    # Find images in both original image folders
    image_lookup = {}

    for folder in SOURCE_DIR.glob("HAM10000_images_part_*"):
        if folder.is_dir():
            for image_path in folder.iterdir():
                if image_path.is_file() and image_path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png"
                }:
                    image_lookup[image_path.stem] = image_path

    print(f"Metadata rows: {len(df)}")
    print(f"Unique image files found: {len(image_lookup)}")

    if not image_lookup:
        print("ERROR: No images found in the image-part folders.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify duplicate image IDs in metadata
    df = df.drop_duplicates(subset=["image_id"], keep="first")

    total_copied = 0

    for class_name in CLASSES:
        class_df = df[
            df["dx"].astype(str).str.lower() == class_name
        ]

        available = []
        missing = 0

        for image_id in class_df["image_id"].astype(str):
            if image_id in image_lookup:
                available.append(image_lookup[image_id])
            else:
                missing += 1

        # Deterministic selection; does not modify source images
        selected = sorted(available, key=lambda p: p.name)[:IMAGES_PER_CLASS]

        target_dir = OUTPUT_DIR / class_name.upper()
        target_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for image_path in selected:
            destination = target_dir / image_path.name

            if not destination.exists():
                shutil.copy2(image_path, destination)

            copied += 1

        total_copied += copied

        print(
            f"{class_name.upper()}: {copied} images | "
            f"available={len(available)} | missing={missing}"
        )

    print("\nDataset preparation completed.")
    print(f"Total selected images: {total_copied}")
    print(f"Output folder: {OUTPUT_DIR}")
    print("\nOriginal dataset was not modified.")


if __name__ == "__main__":
    main()
