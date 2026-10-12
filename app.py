import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================
# Project configuration
# =========================
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "classifier" / "best_model.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "classifier" / "class_names.json"

IMAGE_SIZE = (224, 224)

# This must match the class order used during training.
EXPECTED_CLASSES = [
    "AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"
]

st.set_page_config(
    page_title="SkinSense AI",
    page_icon="🔬",
    layout="centered"
)

st.title("SkinSense AI")
st.subheader("Skin Lesion Classification")
st.write("Upload a skin image to test your trained model.")

st.warning(
    "Research prototype only — not a medical diagnosis."
)


# =========================
# Load trained model
# =========================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return tf.keras.models.load_model(MODEL_PATH)


# =========================
# Load class names
# =========================
@st.cache_data
def load_classes():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # JSON list: ["AKIEC", "BCC", ...]
    if isinstance(data, list):
        classes = data

    elif isinstance(data, dict):
        # JSON dictionary containing a class list
        if isinstance(data.get("class_names"), list):
            classes = data["class_names"]

        elif isinstance(data.get("classes"), list):
            classes = data["classes"]

        # Dictionary: {"0": "AKIEC", "1": "BCC", ...}
        elif all(str(i) in data for i in range(len(data))):
            classes = [data[str(i)] for i in range(len(data))]

        # Dictionary: {"AKIEC": 0, "BCC": 1, ...}
        elif all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in data.values()
        ):
            classes = [
                name
                for name, index in sorted(
                    data.items(), key=lambda item: item[1]
                )
            ]

        else:
            # Do not silently guess an unknown format.
            raise ValueError(
                "Unsupported class_names.json structure. "
                "Check the JSON file contents."
            )
    else:
        raise ValueError("Invalid class_names.json format.")

    classes = [str(name).upper() for name in classes]

    if len(classes) != 7 or set(classes) != set(EXPECTED_CLASSES):
        raise ValueError(
            f"Expected these 7 classes: {EXPECTED_CLASSES}. "
            f"Found: {classes}"
        )

    # Return the order read from the file.
    return classes


# =========================
# Upload image
# =========================
uploaded_file = st.file_uploader(
    "Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Analyzing image..."):
                model = load_model()
                class_names = load_classes()

                output_count = model.output_shape[-1]

                if output_count != len(class_names):
                    raise ValueError(
                        f"Model has {output_count} outputs, "
                        f"but {len(class_names)} class names were loaded."
                    )

                resized_image = image.resize(IMAGE_SIZE)

                image_array = np.asarray(
                    resized_image, dtype=np.float32
                )
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(
                    image_array, verbose=0
                )[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = class_names[predicted_index]
                score = float(predictions[predicted_index]) * 100

            st.divider()
            st.subheader("Prediction Result")

            st.success(f"Predicted Disease: {predicted_class}")
            st.metric("Model Score", f"{score:.2f}%")

            if score < 50:
                st.warning(
                    "Low model score. The prediction may be unreliable."
                )

            st.caption(
                "Model score is not a calibrated medical probability. "
                "Consult a qualified dermatologist for diagnosis."
            )

    except Exception as error:
        st.error(f"Could not analyze image: {error}")

else:
    st.info("Upload an image above to get started.")
