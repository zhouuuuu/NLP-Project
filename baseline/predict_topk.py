"""
predict_topk.py
Load pretrained TF-IDF + Logistic Regression model
and provide a Top-k author prediction interface
with open-set detection (unknown author handling).
"""

import numpy as np
import joblib
from sklearn.preprocessing import normalize as l2norm

# ============================
# 1. Load saved model components
# ============================
VECT_PATH = "models/tfidf_vectorizer.pkl"
CLF_PATH = "models/logistic_model.pkl"
ENC_PATH = "models/label_encoder.pkl"
CEN_PATH = "models/class_centroids.npy"
THR_PATH = "models/open_threshold.txt"

vect = joblib.load(VECT_PATH)
clf = joblib.load(CLF_PATH)
le = joblib.load(ENC_PATH)
Cmat = np.load(CEN_PATH)
with open(THR_PATH, "r") as f:
    threshold = float(f.read().strip())

labels = le.classes_.tolist()

print(f"✅ Model loaded: {len(labels)} known authors | Threshold = {threshold:.3f}")

# ============================
# 2. Prediction function
# ============================
def predict_topk(text, topk=5):
    """
    Input:
        text (str): input text to analyze
        topk (int): number of top similar authors to return

    Output (dict):
        {
          "pred_author": str,
          "unknown": bool,
          "max_cosine_similarity": float,
          "threshold": float,
          "topk": [(author, probability), ...]
        }
    """

    # --- Step 1. TF-IDF encoding ---
    X = vect.transform([text])

    # --- Step 2. Closed-set probability prediction ---
    probs = clf.predict_proba(X)[0]
    order = np.argsort(-probs)
    top_list = [(labels[i], float(probs[i])) for i in order[:topk]]

    # --- Step 3. Compute cosine similarity to class centroids ---
    Xn = l2norm(X, norm="l2")
    sim = float((Xn @ Cmat.T).max())

    # --- Step 4. Open-set detection ---
    is_unknown = sim < threshold
    pred_author = "Unknown Author" if is_unknown else labels[int(order[0])]

    return {
        "pred_author": pred_author,
        "unknown": is_unknown,
        "max_cosine_similarity": sim,
        "threshold": float(threshold),
        "topk": top_list,
    }

# ============================
# 3. Interactive CLI demo
# ============================
if __name__ == "__main__":
    print("\n=== Author Style Prediction (Open-set Aware) ===")
    print("Type a text sample to test (or 'exit' to quit)\n")
    while True:
        text = input("> ").strip()
        if not text or text.lower() == "exit":
            break

        res = predict_topk(text, topk=5)
        print("\n--- Prediction Result ---")
        print(f"Predicted Author : {res['pred_author']}")
        print(f"Unknown Author?  : {res['unknown']}")
        print(f"Max Cosine Sim.  : {res['max_cosine_similarity']:.3f}")
        print(f"Threshold        : {res['threshold']:.3f}")
        print("\nTop-k Similar Authors:")
        for name, score in res["topk"]:
            print(f"  {name:20s} -> {score:.4f}")
        print("\n-------------------------\n")
