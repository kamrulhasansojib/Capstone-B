
import csv
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (128, 128)
THRESHOLD = 0.5
EPSILON = 1e-7

EXPECTED_TEST_IMAGES = 390


# ============================================================
# FILE / FOLDER SELECTORS
# ============================================================

def choose_file(title, filetypes):
    return filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )


def choose_folder(title):
    return filedialog.askdirectory(title=title)


# ============================================================
# IMAGE INDEX
# ============================================================

def build_file_index(folder):
    """
    Index image files by filename, recursively.
    Example: ISIC_0012810.jpg -> Path(...)
    """
    index = {}

    allowed_extensions = {".jpg", ".jpeg", ".png"}

    for path in Path(folder).rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_extensions:
            continue

        index[path.name] = path

    return index


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_rgb(image_path):
    """
    Match the U-Net training preprocessing:
    RGB image resized to 128x128 and scaled to [0, 1].
    """
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        array = np.asarray(
            image,
            dtype=np.float32
        ) / 255.0

    return array


def load_mask(mask_path):
    """
    Ground-truth mask:
    nearest-neighbour resize and binary threshold.
    """
    with Image.open(mask_path) as image:
        image = image.convert("L")

        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.NEAREST
        )

        array = np.asarray(image)

    return (array > 0).astype(np.uint8)


# ============================================================
# DICE AND IOU
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = np.logical_and(
        y_true,
        y_pred
    ).sum()

    union = np.logical_or(
        y_true,
        y_pred
    ).sum()

    dice = (
        2.0 * intersection + EPSILON
    ) / (
        y_true.sum() + y_pred.sum() + EPSILON
    )

    iou = (
        intersection + EPSILON
    ) / (
        union + EPSILON
    )

    return float(dice), float(iou)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    root = Tk()
    root.withdraw()

    try:

        # ----------------------------------------------------
        # 1. Select the fixed split CSV
        # ----------------------------------------------------

        print("Select isic_segmentation_split.csv")

        split_csv = choose_file(
            "Select ISIC segmentation split CSV",
            [("CSV files", "*.csv")]
        )

        if not split_csv:
            print("Cancelled.")
            return

        # ----------------------------------------------------
        # 2. Select original image folder
        # ----------------------------------------------------

        print("Select the original ISIC image folder")

        image_folder = choose_folder(
            "Select ISIC2018_Task1-2_Training_Input"
        )

        if not image_folder:
            print("Cancelled.")
            return

        # ----------------------------------------------------
        # 3. Select original ground-truth masks
        # ----------------------------------------------------

        print("Select the original ISIC ground-truth mask folder")

        mask_folder = choose_folder(
            "Select ISIC2018_Task1_Training_GroundTruth"
        )

        if not mask_folder:
            print("Cancelled.")
            return

        # ----------------------------------------------------
        # 4. Select the trained best model
        # ----------------------------------------------------

        print("Select isic_unet_split_best.keras")

        model_path = choose_file(
            "Select the best trained U-Net model",
            [("Keras model", "*.keras")]
        )

        if not model_path:
            print("Cancelled.")
            return

        # ----------------------------------------------------
        # 5. Load and validate CSV
        # ----------------------------------------------------

        print("\nReading split CSV...")

        df = pd.read_csv(split_csv)

        required_columns = {
            "split",
            "image",
            "mask"
        }

        if not required_columns.issubset(df.columns):
            raise ValueError(
                "The CSV must contain columns: "
                "'split', 'image', 'mask'.\n"
                f"Found columns: {df.columns.tolist()}"
            )

        df["split"] = (
            df["split"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Only reserved test rows are evaluated.
        test_df = df[df["split"] == "test"].copy()

        if test_df.empty:
            raise ValueError(
                "No rows with split='test' were found."
            )

        test_df["image"] = test_df["image"].astype(str).str.strip()
        test_df["mask"] = test_df["mask"].astype(str).str.strip()

        test_df = test_df.drop_duplicates(
            subset=["image", "mask"]
        )

        print(f"CSV rows: {len(df)}")
        print(f"Reserved test pairs: {len(test_df)}")

        if (
            EXPECTED_TEST_IMAGES is not None
            and len(test_df) != EXPECTED_TEST_IMAGES
        ):
            raise ValueError(
                f"Expected {EXPECTED_TEST_IMAGES} test pairs, "
                f"but CSV contains {len(test_df)}. "
                "Check the split CSV before continuing."
            )

        # ----------------------------------------------------
        # 6. Index image and mask files
        # ----------------------------------------------------

        print("\nIndexing original images...")

        image_index = build_file_index(image_folder)

        print(f"Image files indexed: {len(image_index)}")

        print("\nIndexing ground-truth masks...")

        mask_index = build_file_index(mask_folder)

        print(f"Mask files indexed: {len(mask_index)}")

        missing_images = []
        missing_masks = []

        for row in test_df.itertuples(index=False):

            image_name = Path(row.image).name
            mask_name = Path(row.mask).name

            if image_name not in image_index:
                missing_images.append(image_name)

            if mask_name not in mask_index:
                missing_masks.append(mask_name)

        if missing_images or missing_masks:
            raise FileNotFoundError(
                f"Missing test images: {len(missing_images)}\n"
                f"Examples: {missing_images[:5]}\n\n"
                f"Missing test masks: {len(missing_masks)}\n"
                f"Examples: {missing_masks[:5]}\n\n"
                "Verify the selected folders and split CSV."
            )

        print("All reserved test images and masks found.")

        # ----------------------------------------------------
        # 7. Load model
        # ----------------------------------------------------

        print("\nLoading trained U-Net model...")

        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print("Model loaded successfully.")

        # ----------------------------------------------------
        # 8. Evaluate only test set
        # ----------------------------------------------------

        results = []

        print("\nStarting reserved test evaluation...")

        for number, row in enumerate(
            test_df.itertuples(index=False),
            start=1
        ):

            image_name = Path(row.image).name
            mask_name = Path(row.mask).name

            image_path = image_index[image_name]
            mask_path = mask_index[mask_name]

            image_array = load_rgb(image_path)
            true_mask = load_mask(mask_path)

            prediction = model.predict(
                np.expand_dims(image_array, axis=0),
                verbose=0
            )

            # This evaluator expects a single-channel mask output.
            if (
                prediction.ndim != 4
                or prediction.shape[-1] != 1
            ):
                raise ValueError(
                    "Unexpected model output shape: "
                    f"{prediction.shape}. Expected "
                    "(batch, height, width, 1)."
                )

            predicted_mask = (
                prediction[0, :, :, 0] >= THRESHOLD
            ).astype(np.uint8)

            dice, iou = calculate_metrics(
                true_mask,
                predicted_mask
            )

            results.append({
                "image_id": Path(image_name).stem,
                "dice": dice,
                "iou": iou
            })

            if (
                number % 25 == 0
                or number == len(test_df)
            ):
                print(
                    f"Processed {number}/{len(test_df)} "
                    f"| Dice: {dice:.4f} "
                    f"| IoU: {iou:.4f}"
                )

        # ----------------------------------------------------
        # 9. Final metrics
        # ----------------------------------------------------

        results_df = pd.DataFrame(results)

        mean_dice = results_df["dice"].mean()
        mean_iou = results_df["iou"].mean()

        median_dice = results_df["dice"].median()
        median_iou = results_df["iou"].median()

        print("\n" + "=" * 50)
        print("ISIC 2018 SEGMENTATION TEST RESULTS")
        print("=" * 50)

        print(f"Test images evaluated: {len(results_df)}")

        print(
            f"Mean Dice: {mean_dice:.4f} "
            f"({mean_dice * 100:.2f}%)"
        )

        print(
            f"Mean IoU: {mean_iou:.4f} "
            f"({mean_iou * 100:.2f}%)"
        )

        print(
            f"Median Dice: {median_dice:.4f} "
            f"({median_dice * 100:.2f}%)"
        )

        print(
            f"Median IoU: {median_iou:.4f} "
            f"({median_iou * 100:.2f}%)"
        )

        print("\nLowest 10 Dice scores:")

        print(
            results_df
            .sort_values("dice")
            .head(10)
            .to_string(index=False)
        )

        # ----------------------------------------------------
        # 10. Save results
        # ----------------------------------------------------

        save_path = filedialog.asksaveasfilename(
            title="Save segmentation test results",
            defaultextension=".csv",
            initialfile="isic_segmentation_test_results.csv",
            filetypes=[("CSV files", "*.csv")]
        )

        if save_path:

            results_df.to_csv(
                save_path,
                index=False
            )

            summary_path = Path(save_path).with_name(
                "isic_segmentation_test_summary.txt"
            )

            with open(
                summary_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(
                    "ISIC 2018 Segmentation Test Evaluation\n"
                )

                file.write(
                    f"Model: {Path(model_path).name}\n"
                )

                file.write(
                    f"Test images: {len(results_df)}\n"
                )

                file.write(
                    f"Mean Dice: {mean_dice:.6f} "
                    f"({mean_dice * 100:.2f}%)\n"
                )

                file.write(
                    f"Mean IoU: {mean_iou:.6f} "
                    f"({mean_iou * 100:.2f}%)\n"
                )

                file.write(
                    f"Median Dice: {median_dice:.6f}\n"
                )

                file.write(
                    f"Median IoU: {median_iou:.6f}\n"
                )

            print(f"\nResults saved: {save_path}")
            print(f"Summary saved: {summary_path}")

        messagebox.showinfo(
            "Evaluation completed",
            f"Test images: {len(results_df)}\n"
            f"Mean Dice: {mean_dice * 100:.2f}%\n"
            f"Mean IoU: {mean_iou * 100:.2f}%"
        )

    except Exception as error:

        print(f"\nERROR: {error}")

        messagebox.showerror(
            "Evaluation failed",
            str(error)
        )

        raise

    finally:
        root.destroy()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
