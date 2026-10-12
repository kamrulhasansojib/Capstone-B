
import sys
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "classifier" / "best_model.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "classifier" / "class_names.json"

IMAGE_SIZE = (224, 224)


def predict_skin_disease(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    score = float(predictions[predicted_index]) * 100

    print("\nSkinSense AI Prediction")
    print("-" * 30)
    print(f"Predicted Disease: {predicted_class}")
    print(f"Model Score: {score:.2f}%")

    if score < 50:
        print("Note: Prediction confidence is low.")
    
    print("\nThis prototype is not a medical diagnosis.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an image path.")
        print('Example: python ai_models/classification/predict.py "F:\\SkinSense-AI\\Small HAM\\BCC\\ISIC_0024332.jpg"')
    else:
        predict_skin_disease(sys.argv[1])
