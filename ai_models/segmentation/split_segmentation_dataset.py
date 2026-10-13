import csv
import random
from pathlib import Path
from tkinter import Tk, filedialog

SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15


def select_folder(title):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()

    if not folder:
        raise SystemExit("Folder selection cancelled.")

    return Path(folder)


def main():
    print("Select ISIC Task 1 input images folder.")
    image_dir = select_folder("Select input images folder")

    print("Select ISIC Task 1 ground-truth masks folder.")
    mask_dir = select_folder("Select ground-truth masks folder")

    pairs = []

    for image_path in sorted(image_dir.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}
            or not image_path.stem.startswith("ISIC_")
        ):
            continue

        mask_path = mask_dir / f"{image_path.stem}_segmentation.png"

        if mask_path.is_file():
            pairs.append((image_path.name, mask_path.name))

    if len(pairs) < 3:
        raise ValueError("Not enough matching image-mask pairs.")

    random.Random(SEED).shuffle(pairs)

    total = len(pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VALIDATION_RATIO)

    splits = {
        "train": pairs[:train_end],
        "validation": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    # Store paths relative to the selected dataset folders.
    output_dir = Path(__file__).resolve().parent / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "isic_segmentation_split.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "image", "mask"])

        for split_name, split_pairs in splits.items():
            for image_name, mask_name in split_pairs:
                writer.writerow([
                    split_name,
                    image_name,
                    mask_name,
                ])

    # Check that every pair appears in exactly one split.
    all_images = [
        image_name
        for split_pairs in splits.values()
        for image_name, _ in split_pairs
    ]

    print("\n--- Segmentation Split Report ---")
    print(f"Total matched pairs: {total}")
    print(f"Training: {len(splits['train'])}")
    print(f"Validation: {len(splits['validation'])}")
    print(f"Test: {len(splits['test'])}")
    print(f"Unique image IDs: {len(set(all_images))}")
    print(f"Split CSV saved: {csv_path}")

    if len(set(all_images)) != total:
        raise RuntimeError("Duplicate or missing image IDs detected.")

    print("Split creation completed successfully.")


if __name__ == "__main__":
    main()
