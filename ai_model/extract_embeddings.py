# ai_model/extract_embeddings.py
import os
import pickle
from mobilenet import get_embedding

dataset_dir = "ai_model/dataset"
embeddings = {}

for product_id in os.listdir(dataset_dir):
    product_folder = os.path.join(dataset_dir, product_id)
    if not os.path.isdir(product_folder):
        continue

    embeddings[product_id] = []
    for img_file in os.listdir(product_folder):
        img_path = os.path.join(product_folder, img_file)
        try:
            emb = get_embedding(img_path)
            embeddings[product_id].append(emb)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Save embeddings to disk
with open("ai_model/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Embeddings saved to ai_model/embeddings.pkl")
