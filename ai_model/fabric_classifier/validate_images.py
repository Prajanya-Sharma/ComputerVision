import os
from PIL import Image

dataset_dir = "ai_model/fabric_classifier/dataset"
checked = 0
corrupted = 0

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        file_path = os.path.join(root, file)
        checked += 1
        try:
            img = Image.open(file_path)
            img.verify()
        except Exception as e:
            print(f"❌ Corrupted image: {file_path} — {e}")
            corrupted += 1

print(f"\n✅ Checked: {checked} images")
print(f"❗ Found {corrupted} corrupted images")
