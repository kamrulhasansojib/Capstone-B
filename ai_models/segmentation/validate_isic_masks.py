from pathlib import Path
from tkinter import Tk, filedialog
from PIL import Image
import numpy as np


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
    print("Select the Task 1 INPUT images folder.")
    image_dir = select_folder("Select ISIC Task 1 Input Images")

    print("Select the Task 1 GROUND-TRUTH masks folder.")
    mask_dir = select_folder("Select ISIC Task 1 GroundTruth Masks")

    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and p.stem.startswith("ISIC_")
    )

    checked = 0
    mismatches = []
    empty_masks = []
    read_errors = []
    mask_values = set()

    for image_path in images:
        mask_path = mask_dir / f"{image_path.stem}_segmentation.png"

        try:
            with Image.open(image_path) as image:
                image_size = image.size

            with Image.open(mask_path) as mask:
                array = np.array(mask.convert("L"))
                mask_size = mask.size

            checked += 1
            mask_values.update(np.unique(array).tolist())

            if image_size != mask_size:
                mismatches.append(image_path.name)

            if not np.any(array > 0):
                empty_masks.append(mask_path.name)

        except Exception as exc:
            read_errors.append((image_path.name, str(exc)))

    print("\n--- ISIC Task 1 Validation Report ---")
    print(f"Input images found: {len(images)}")
    print(f"Images checked: {checked}")
    print(f"Dimension mismatches: {len(mismatches)}")
    print(f"Empty masks: {len(empty_masks)}")
    print(f"Read errors: {len(read_errors)}")
    print(f"Mask pixel values: {sorted(mask_values)}")

    if mismatches:
        print("First dimension mismatches:", mismatches[:10])

    if empty_masks:
        print("First empty masks:", empty_masks[:10])

    if read_errors:
        print("First read errors:", read_errors[:10])


if __name__ == "__main__":
    main()
