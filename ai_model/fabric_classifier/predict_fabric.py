import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import sys

# Paths
MODEL_PATH = 'ai_model/fabric_classifier/fabric_model.keras'
DATASET_DIR = 'ai_model/fabric_classifier/dataset'

# Globals to avoid reloading when imported
model = None
class_names = []

def load_model_and_classes():
    """Loads the model and class names only once."""
    global model, class_names
    if model is None:
        print("🧠 Loading model and class names...")
        model = load_model(MODEL_PATH)
        class_names = sorted([
            d for d in os.listdir(DATASET_DIR)
            if os.path.isdir(os.path.join(DATASET_DIR, d))
        ])

def load_and_prepare_image(img_path):
    """Loads and preprocesses an image for prediction."""
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        raise RuntimeError(f"❌ Error loading image {img_path}: {e}")

def predict_top_n(img_path, n=3):
    """Predicts the top-n classes for the given image."""
    load_model_and_classes()
    img_array = load_and_prepare_image(img_path)
    preds = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(preds)[-n:][::-1]
    return [(class_names[i], preds[i]) for i in top_indices]

def run_prediction(folder, image_name):
    """Runs prediction and prints results."""
    img_path = os.path.join(folder, image_name)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"❌ Image file not found: {img_path}")

    print(f"🔎 Running prediction on: {img_path}")
    top_results = predict_top_n(img_path, n=3)

    print("\n🧵 Top Fabric Predictions:")
    for i, (label, confidence) in enumerate(top_results, start=1):
        print(f"{i}. {label} — Confidence: {confidence:.4f}")

    print(f"\n✅ Final Prediction: {top_results[0][0]} (confidence: {top_results[0][1]:.4f})")

    return top_results[0][0]  # Return top prediction

def main():
    if len(sys.argv) != 3:
        print("❗ Usage: python predict_fabric.py <folder_path> <image_filename>")
        sys.exit(1)

    folder = sys.argv[1]
    image_name = sys.argv[2]

    try:
        run_prediction(folder, image_name)
    except Exception as e:
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
