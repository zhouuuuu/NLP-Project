"""
predict_topk.py
Load pretrained TF-IDF + Logistic Regression model
and provide a Top-k author similarity prediction interface.
"""

import numpy as np, joblib
from sklearn.preprocessing import normalize as l2norm

# ========== Load saved models ==========
vect = joblib.load("models/tfidf_vectorizer.pkl")
clf  = joblib.load("models/logistic_model.pkl")
le   = joblib.load("models/label_encoder.pkl")
Cmat = np.load("models/class_centroids.npy")
with open("models/open_threshold.txt") as f:
    threshold = float(f.read().strip())
labels = le.classes_.tolist()

# ========== Prediction interface ==========
def predict_topk(text, topk=5):
    """
    Input a text snippet → return predicted author,
    open-set flag, cosine similarity, and Top-k probabilities.
    """
    X = vect.transform([text])
    probs = clf.predict_proba(X)[0]
    order = np.argsort(-probs)
    top_list = [(labels[i], float(probs[i])) for i in order[:topk]]

    Xn = l2norm(X, norm="l2")
    sim = float((Xn @ Cmat.T).max())
    is_unknown = sim < threshold
    pred_author = "Unknown Author" if is_unknown else labels[int(order[0])]

    return {
        "pred_author": pred_author,
        "unknown": is_unknown,
        "max_cosine_similarity": sim,
        "threshold": float(threshold),
        "topk": top_list
    }

# ========== Example CLI usage ==========
if __name__ == "__main__":
    sample_text = input("Enter a text snippet:\n> ")
    res = predict_topk(sample_text, topk=5)
    print("\nPredicted author:", res["pred_author"], "| Unknown:", res["unknown"])
    print("Max cosine similarity:", round(res["max_cosine_similarity"], 3))
    print("Top-k similar authors:")
    for name, score in res["topk"]:
        print(f"  {name:20s} -> {score:.4f}")
