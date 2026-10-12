
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(r"F:\SkinSense-AI")

AUDIT_CSV = (
    PROJECT_ROOT / "data" / "splits"
    / "small_ham_metadata_audit.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT / "data" / "splits"
    / "lesion_level_split.csv"
)

SEED = 42
N_CANDIDATES = 200

TARGETS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def make_candidate(df, seed):
    # Split off approximately 30% for validation + test.
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed,
    )

    train_idx, temp_idx = next(
        outer.split(df, groups=df["lesion_id"])
    )

    train = df.iloc[train_idx].copy()
    temp = df.iloc[temp_idx].copy()

    # Divide the remaining 30% equally.
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed + 10000,
    )

    val_idx, test_idx = next(
        inner.split(temp, groups=temp["lesion_id"])
    )

    validation = temp.iloc[val_idx].copy()
    test = temp.iloc[test_idx].copy()

    return train, validation, test


def candidate_score(parts, full_df):
    total = len(full_df)
    overall = full_df["folder_class"].value_counts(normalize=True)

    score = 0.0

    for name, part in parts.items():
        target_size = TARGETS[name] * total

        # Penalize deviation from target split size.
        size_error = abs(len(part) - target_size) / total
        score += 2.0 * size_error

        # Penalize class-distribution deviation.
        proportions = part["folder_class"].value_counts(
            normalize=True
        )

        all_classes = overall.index

        for class_name in all_classes:
            actual = proportions.get(class_name, 0.0)
            expected = overall[class_name]
            score += abs(actual - expected)

        # Missing classes are a strong warning.
        missing_count = len(set(all_classes) - set(part["folder_class"]))
        score += 10.0 * missing_count

    return score


def main():
    df = pd.read_csv(AUDIT_CSV)

    required = {
        "image_id", "path", "folder_class", "lesion_id"
    }

    if not required.issubset(df.columns):
        raise ValueError("Audit CSV is missing required columns.")

    if df["lesion_id"].isna().any():
        raise ValueError("Missing lesion IDs found.")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs found.")

    best_parts = None
    best_score = float("inf")

    # Try multiple deterministic candidates and select the best.
    for seed in range(SEED, SEED + N_CANDIDATES):
        parts = dict(zip(
            ["train", "validation", "test"],
            make_candidate(df, seed),
        ))

        if any(part.empty for part in parts.values()):
            continue

        # Verify all classes appear in every split.
        expected_classes = set(df["folder_class"].unique())

        if any(
            set(part["folder_class"]) != expected_classes
            for part in parts.values()
        ):
            continue

        score = candidate_score(parts, df)

        if score < best_score:
            best_score = score
            best_parts = parts

    if best_parts is None:
        raise RuntimeError(
            "Could not find a split containing every class. "
            "Review the dataset and reduce split constraints."
        )

    # Combine the selected split.
    result_parts = []

    for split_name, part in best_parts.items():
        part = part.copy()
        part["split"] = split_name
        result_parts.append(part)

    result = pd.concat(result_parts, ignore_index=True)

    # Critical leakage check.
    split_count = result.groupby("lesion_id")["split"].nunique()

    if (split_count > 1).any():
        raise RuntimeError("Lesion leakage detected!")

    if len(result) != len(df):
        raise RuntimeError("Some images were lost or duplicated.")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)

    print("=" * 60)
    print("SkinSense AI - Lesion-Level Split Audit")
    print("=" * 60)
    print(f"Total images: {len(result)}")
    print(f"Unique lesions: {result['lesion_id'].nunique()}")
    print(f"Candidate score: {best_score:.4f}")

    for split_name in TARGETS:
        part = result[result["split"] == split_name]

        print(f"\n{split_name.upper()}")
        print(f"Images: {len(part)} ({len(part)/len(result):.1%})")
        print(f"Unique lesions: {part['lesion_id'].nunique()}")
        print(
            part["folder_class"]
            .value_counts()
            .reindex(sorted(result["folder_class"].unique()),
                     fill_value=0)
            .to_string()
        )

    print("\nPASS: No lesion crosses split boundaries.")
    print("Saved:", OUTPUT_CSV)


if __name__ == "__main__":
    main()
