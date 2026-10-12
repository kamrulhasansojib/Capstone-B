
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"F:\SkinSense-AI")
DATASET_DIR = PROJECT_ROOT / "Small HAM"

METADATA_PATH = Path(
    r"C:\Users\HELLO\OneDrive\Desktop\Skin Diseases"
    r"\HAM10000\HAM10000_metadata.csv"
)

CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

DX_TO_CLASS = {
    "akiec": "AKIEC",
    "bcc": "BCC",
    "bkl": "BKL",
    "df": "DF",
    "mel": "MEL",
    "nv": "NV",
    "vasc": "VASC",
}

def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    if not DATASET_DIR.exists():
        raise FileNotFoundError(DATASET_DIR)

    metadata = pd.read_csv(METADATA_PATH)
    metadata["class_name"] = metadata["dx"].map(DX_TO_CLASS)

    image_to_metadata = metadata.set_index("image_id").to_dict("index")

    rows = []
    unmatched = []
    label_mismatches = []

    for class_name in CLASSES:
        class_dir = DATASET_DIR / class_name

        if not class_dir.is_dir():
            print(f"Missing class folder: {class_dir}")
            continue

        for image_path in class_dir.iterdir():
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in {
                ".jpg", ".jpeg", ".png"
            }:
                continue

            image_id = image_path.stem

            # Supports filenames such as ISIC_0027937.jpg.
            if image_id not in image_to_metadata:
                unmatched.append(str(image_path))
                continue

            record = image_to_metadata[image_id]

            if record["class_name"] != class_name:
                label_mismatches.append({
                    "image": image_id,
                    "folder_class": class_name,
                    "metadata_class": record["class_name"],
                })

            rows.append({
                "image_id": image_id,
                "path": str(image_path.resolve()),
                "folder_class": class_name,
                "metadata_class": record["class_name"],
                "lesion_id": record["lesion_id"],
            })

    result = pd.DataFrame(rows)

    print("=" * 60)
    print("Small HAM Metadata Matching Audit")
    print("=" * 60)
    print("Matched images:", len(result))
    print("Unmatched images:", len(unmatched))
    print("Folder/metadata label mismatches:", len(label_mismatches))

    if not result.empty:
        print("\nMatched class counts:")
        print(result["folder_class"].value_counts().sort_index())

        print("\nUnique lesion IDs:", result["lesion_id"].nunique())

        print("\nImages per lesion:")
        print(result.groupby("lesion_id").size().value_counts().sort_index())

        output_path = (
            PROJECT_ROOT / "data" / "splits"
            / "small_ham_metadata_audit.csv"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print("\nAudit CSV saved:", output_path)

    if unmatched:
        print("\nFirst unmatched images:")
        for path in unmatched[:10]:
            print(path)

    if label_mismatches:
        print("\nFirst label mismatches:")
        for item in label_mismatches[:10]:
            print(item)

    if unmatched or label_mismatches:
        print(
            "\nWARNING: Resolve unmatched files or label mismatches "
            "before creating final splits."
        )
    else:
        print("\nAll discovered images matched metadata and labels.")

if __name__ == "__main__":
    main()
