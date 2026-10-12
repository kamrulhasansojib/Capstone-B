
from pathlib import Path
from PIL import Image

# Project and dataset paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_DIR / "Small HAM"

CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def check_dataset():
    print("=" * 50)
    print("SkinSense AI - Dataset Check")
    print("=" * 50)

    if not DATASET_DIR.exists():
        print(f"Dataset not found: {DATASET_DIR}")
        return

    total_images = 0
    valid_images = 0

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name

        if not class_dir.exists():
            print(f"{class_name}: Folder missing!")
            continue

        images = [
            path for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        class_valid = 0

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
                class_valid += 1
            except Exception as error:
                print(f"Invalid image: {image_path.name}")

        total_images += len(images)
        valid_images += class_valid

        print(
            f"{class_name}: "
            f"{len(images)} images, "
            f"{class_valid} valid"
        )

    print("-" * 50)
    print(f"Total images found: {total_images}")
    print(f"Valid images: {valid_images}")
    print("=" * 50)

if __name__ == "__main__":
    check_dataset()
