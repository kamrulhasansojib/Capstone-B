from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import numpy as np
import tensorflow as tf
from data_loader import load_datasets

MODEL_PATH = r"F:\SkinSense-AI\models\classifier\best_model.keras"

# Load validation dataset
_, val_ds = load_datasets()

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

class_names = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]

print("\n===== SkinSense AI Evaluation =====")
print(f"Validation Images: {len(y_true)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

print("\n===== Class-wise Report =====")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("===== Confusion Matrix =====")
print("Class order:", class_names)
print(confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
))
