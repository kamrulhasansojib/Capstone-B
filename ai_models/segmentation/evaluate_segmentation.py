
import random
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


IMAGE_SIZE = (128, 128)
SEED = 42


def select_folder(title):
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()

    if not folder:
        raise SystemExit("Folder selection cancelled.")

    return Path(folder)


def load_pair(image_path, mask_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            IMAGE_SIZE, Image.Resampling.BILINEAR
        )
        image = np.asarray(image, dtype=np.float32) / 255.0

    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        mask = mask.resize(
            IMAGE_SIZE, Image.Resampling.NEAREST
        )
        mask = np.asarray(mask, dtype=np.uint8)
        mask = (mask >= 128).astype(np.uint8)

    return image, mask


def calculate_metrics(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    dice = (
        2.0 * intersection / (y_true.sum() + y_pred.sum())
        if y_true.sum() + y_pred.sum() > 0
        else 1.0
    )

    iou = intersection / union if union > 0 else 1.0

    return float(dice), float(iou)


def main():
    project_root = Path(__file__).resolve().parents[2]
    default_model = (
        project_root / "models" / "segmentation"
        / "isic_unet_best.keras"
    )

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    model_file = filedialog.askopenfilename(
        title="Select the best trained U-Net model",
        initialdir=str(default_model.parent),
        filetypes=[("Keras model", "*.keras")],
    )
    root.destroy()

    if not model_file:
        raise SystemExit("Model selection cancelled.")

    image_dir = select_folder("Select ISIC Task 1 input images")
    mask_dir = select_folder("Select ISIC Task 1 ground-truth masks")

    model = tf.keras.models.load_model(
        model_file,
        custom_objects={
            "dice_loss": lambda y_true, y_pred: tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_true, y_pred)
            )
        },
        compile=False,
    )

    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and p.stem.startswith("ISIC_")
        and (mask_dir / f"{p.stem}_segmentation.png").is_file()
    )

    if not image_files:
        raise SystemExit("No matching image-mask pairs found.")

    random.Random(SEED).shuffle(image_files)

    # This is a visual sample, not independent test evaluation.
    sample_files = image_files[:min(12, len(image_files))]

    results = []

    for image_path in sample_files:
        mask_path = mask_dir / f"{image_path.stem}_segmentation.png"

        image, true_mask = load_pair(image_path, mask_path)
        probability = model.predict(
            np.expand_dims(image, axis=0), verbose=0
        )[0, :, :, 0]

        predicted_mask = (probability >= 0.5).astype(np.uint8)
        dice, iou = calculate_metrics(true_mask, predicted_mask)

        results.append({
            "name": image_path.name,
            "image": image,
            "true_mask": true_mask,
            "predicted_mask": predicted_mask,
            "dice": dice,
            "iou": iou,
        })

    output_dir = project_root / "models" / "segmentation" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, result in enumerate(results, start=1):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(result["image"])
        axes[0].set_title("Original Image")

        axes[1].imshow(result["true_mask"], cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")

        axes[2].imshow(
            result["predicted_mask"], cmap="gray", vmin=0, vmax=1
        )
        axes[2].set_title(
            f"Predicted Mask\nDice: {result['dice']:.3f}"
            f" | IoU: {result['iou']:.3f}"
        )

        for axis in axes:
            axis.axis("off")

        fig.suptitle(result["name"])
        fig.tight_layout()

        output_path = output_dir / f"prediction_{index:02d}.png"
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(
            f"{result['name']} | "
            f"Dice: {result['dice']:.4f} | "
            f"IoU: {result['iou']:.4f}"
        )

    mean_dice = float(np.mean([r["dice"] for r in results]))
    mean_iou = float(np.mean([r["iou"] for r in results]))

    print("\n--- Sample Visualization Summary ---")
    print(f"Images visualized: {len(results)}")
    print(f"Mean sample Dice: {mean_dice:.4f}")
    print(f"Mean sample IoU: {mean_iou:.4f}")
    print(f"Saved visualizations: {output_dir}")
    print("Note: These sample scores are not independent test metrics.")


if __name__ == "__main__":
    main()
