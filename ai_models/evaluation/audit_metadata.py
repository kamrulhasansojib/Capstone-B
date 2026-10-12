
from pathlib import Path
import pandas as pd

CSV_PATH = Path(
    r"C:\Users\HELLO\OneDrive\Desktop\Skin Diseases"
    r"\HAM10000\HAM10000_metadata.csv"
)

def main():
    print("=" * 55)
    print("SkinSense AI - HAM10000 Metadata Audit")
    print("=" * 55)

    if not CSV_PATH.exists():
        print("ERROR: Metadata CSV not found.")
        print("Check the filename and folder path.")
        return

    df = pd.read_csv(CSV_PATH)

    print(f"Total metadata rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(df.columns)

    if missing:
        print(f"Missing required columns: {sorted(missing)}")
        return

    print("\nDisease class counts:")
    print(df["dx"].value_counts().sort_index().to_string())

    print("\nUnique image IDs:", df["image_id"].nunique())
    print("Unique lesion IDs:", df["lesion_id"].nunique())

    duplicates = df["image_id"].duplicated().sum()
    print("Duplicate image IDs:", duplicates)

    lesion_counts = df.groupby("lesion_id")["image_id"].nunique()

    repeated = lesion_counts[lesion_counts > 1]

    print("Lesions with multiple images:", len(repeated))
    print(
        "Images belonging to repeated lesions:",
        int(lesion_counts[lesion_counts > 1].sum())
    )

    print("\nMetadata audit completed.")

if __name__ == "__main__":
    main()
