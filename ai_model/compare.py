# ai_model/compare.py
import pickle
import numpy as np
from mobilenet import get_embedding
from sklearn.metrics.pairwise import cosine_similarity

def find_best_match(upload_path, threshold=0.75):
    with open("ai_model/embeddings.pkl", "rb") as f:
        known_embeddings = pickle.load(f)

    upload_emb = get_embedding(upload_path).reshape(1, -1)

    best_score = -1
    best_product = None

    for product_id, emb_list in known_embeddings.items():
        for emb in emb_list:
            emb = np.array(emb).reshape(1, -1)
            score = cosine_similarity(upload_emb, emb)[0][0]
            if score > best_score:
                best_score = score
                best_product = product_id

    match = best_score >= threshold
    return {
        "match": match,
        "product_id": best_product if match else None,
        "confidence": round(float(best_score), 4)
    }
