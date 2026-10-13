
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ---------------- Configuration ----------------

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SEGMENTATION_DIR = MODEL_DIR / "segmentation"

CLASSIFIER_PATH = MODEL_DIR / "skinsense_best.keras"

SEGMENTATION_CANDIDATES = [
    SEGMENTATION_DIR / "isic_unet_split_best.keras",
    SEGMENTATION_DIR / "isic_unet_best.keras",
]

CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

CLASSIFICATION_SIZE = (224, 224)
SEGMENTATION_SIZE = (128, 128)


# ---------------- Streamlit page ----------------

st.set_page_config(
    page_title="SkinSense AI",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 SkinSense AI")
st.write("AI-assisted skin lesion analysis")

st.warning(
    "Research prototype only. Predictions are not a medical diagnosis. "
    "Consult a qualified healthcare professional for medical assessment."
)


# ---------------- Model loading ----------------

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(
        str(model_path),
        compile=False,
    )


def get_segmentation_model_path():
    for path in SEGMENTATION_CANDIDATES:
        if path.is_file():
            return path
    return None


# ---------------- Image preprocessing ----------------

def prepare_classification_image(image):
    image = image.convert("RGB").resize(
        CLASSIFICATION_SIZE,
        Image.Resampling.BILINEAR,
    )
    # EfficientNetB0's Keras implementation includes its own
    # rescaling layer; supply RGB pixels in the 0-255 range.
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def prepare_segmentation_image(image):
    image = image.convert("RGB").resize(
        SEGMENTATION_SIZE,
        Image.Resampling.BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


# ---------------- User interface ----------------

tab_classification, tab_segmentation = st.tabs(
    ["🧪 Disease Classification", "🎯 Lesion Segmentation"]
)


with tab_classification:
    st.header("Skin Disease Classification")
    st.write("Upload a dermoscopic image for a 7-class prediction.")

    classifier_upload = st.file_uploader(
        "Choose a skin image",
        type=["jpg", "jpeg", "png"],
        key="classification_upload",
    )

    if classifier_upload is not None:
        image = Image.open(classifier_upload).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            width=400,
        )

        if st.button(
            "Classify Image",
            type="primary",
            key="classify_button",
        ):
            if not CLASSIFIER_PATH.is_file():
                st.error(
                    "Classifier model not found. "
                    "Check models/skinsense_best.keras."
                )
            else:
                try:
                    with st.spinner("Running classification..."):
                        classifier = load_model(CLASSIFIER_PATH)
                        input_array = prepare_classification_image(image)

                        prediction = classifier.predict(
                            input_array,
                            verbose=0,
                        )[0]

                        if len(prediction) != len(CLASS_NAMES):
                            raise ValueError(
                                f"Model returned {len(prediction)} scores, "
                                f"but {len(CLASS_NAMES)} classes are configured."
                            )

                        index = int(np.argmax(prediction))
                        predicted_class = CLASS_NAMES[index]
                        score = float(prediction[index]) * 100.0

                    st.success(f"Predicted class: {predicted_class}")
                    st.metric(
                        "Model score",
                        f"{score:.2f}%",
                    )

                    st.caption(
                        "The model score is not a calibrated probability "
                        "and should not be interpreted as diagnostic certainty."
                    )

                except Exception as error:
                    st.error(f"Could not classify image: {error}")


with tab_segmentation:
    st.header("Skin Lesion Segmentation")
    st.write(
        "Upload an image to visualize the region predicted as a lesion."
    )

    segmentation_upload = st.file_uploader(
        "Choose a skin image",
        type=["jpg", "jpeg", "png"],
        key="segmentation_upload",
    )

    threshold = st.slider(
        "Mask threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        key="segmentation_threshold",
    )

    if segmentation_upload is not None:
        image = Image.open(segmentation_upload).convert("RGB")

        if st.button(
            "Segment Lesion",
            type="primary",
            key="segment_button",
        ):
            segmentation_path = get_segmentation_model_path()

            if segmentation_path is None:
                st.error(
                    "No segmentation model found. "
                    "Finish training or check models/segmentation."
                )
            else:
                try:
                    with st.spinner("Generating segmentation mask..."):
                        segmenter = load_model(segmentation_path)
                        input_array = prepare_segmentation_image(image)

                        probabilities = segmenter.predict(
                            input_array,
                            verbose=0,
                        )[0, :, :, 0]

                        binary_mask = (
                            probabilities >= threshold
                        ).astype(np.uint8) * 255

                        mask_image = Image.fromarray(binary_mask).resize(
                            image.size,
                            Image.Resampling.NEAREST,
                        )

                        mask_array = np.asarray(mask_image) > 0

                        original_array = np.asarray(image).copy()
                        overlay = original_array.copy()

                        red = np.array([255, 40, 40], dtype=np.float32)
                        overlay[mask_array] = (
                            0.60 * original_array[mask_array]
                            + 0.40 * red
                        ).astype(np.uint8)

                        overlay_image = Image.fromarray(overlay)
                        mask_area = float(mask_array.mean() * 100.0)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.image(
                            image,
                            caption="Original image",
                            use_container_width=True,
                        )

                    with col2:
                        st.image(
                            mask_image,
                            caption="Predicted mask",
                            use_container_width=True,
                        )

                    with col3:
                        st.image(
                            overlay_image,
                            caption="Red overlay",
                            use_container_width=True,
                        )

                    st.metric(
                        "Predicted mask area",
                        f"{mask_area:.2f}%",
                    )

                    st.caption(
                        f"Segmentation model: {segmentation_path.name}. "
                        "Mask area is not a disease risk score."
                    )

                except Exception as error:
                    st.error(f"Could not segment image: {error}")
