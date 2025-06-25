import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = os.path.join(os.path.dirname(__file__),"fabric_model.h5")
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else None

if not IMAGE_PATH or not os.path.exists(IMAGE_PATH):
    print("❌ Please provide a valid image path.")
    sys.exit(1)

model = load_model(MODEL_PATH)

img = image.load_img(IMAGE_PATH, target_size=(224, 224))  # Adjust if your model used a different size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])

# Get class labels from the folder structure
dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')
class_labels = sorted(os.listdir(dataset_path))

# Output result
print(f"\n🧵 Predicted Fabric Type: {class_labels[predicted_class_index]}")
print(f"📊 Confidence: {predictions[0][predicted_class_index]:.4f}")