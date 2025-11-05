"""
train_baseline.py
Train a TF-IDF + Logistic Regression author-style classifier
and save all necessary model artifacts for later inference.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, os, random, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# 1. Load dataset

df = pd.read_csv("../data/author_style_dataset_OPENSET.csv")
text_col, author_col, split_col = "text", "author", "split"
assert all(col in df.columns for col in [text_col, author_col, split_col]), \
    "Dataset must contain columns: text, author, split"


df[text_col] = df[text_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df[text_col].str.len() > 0]


# 2. Use pre-defined splits

train_df = df[df[split_col] == "train"].copy()
val_df   = df[df[split_col] == "val"].copy()
test_df  = df[df[split_col] == "test"].copy()

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Identify known and unknown authors
known_authors = sorted(train_df[author_col].unique())
unknown_authors = sorted(set(test_df[author_col]) - set(known_authors))
print(f"Known authors: {len(known_authors)} | Unknown authors in test: {len(unknown_authors)}")

# For open-set evaluation
df_known_test = test_df[test_df[author_col].isin(known_authors)]
df_unknown_test = test_df[test_df[author_col].isin(unknown_authors)]


# 3. TF-IDF feature extraction

vect = TfidfVectorizer(analyzer="char", ngram_range=(1,5),
                       min_df=3, sublinear_tf=True, norm="l2", lowercase=False)
X_train = vect.fit_transform(train_df[text_col])
X_val   = vect.transform(val_df[text_col])
X_test  = vect.transform(df_known_test[text_col])
X_unk   = vect.transform(df_unknown_test[text_col])

le = LabelEncoder().fit(train_df[author_col])
y_train = le.transform(train_df[author_col])
y_val   = le.transform(val_df[author_col])
y_test  = le.transform(df_known_test[author_col])
labels = le.classes_.tolist()


# 4. Train Logistic Regression model

clf = LogisticRegression(max_iter=3000, C=2.0, solver="saga",
                         multi_class="multinomial", n_jobs=-1, random_state=RANDOM_SEED)
clf.fit(X_train, y_train)


# 5. Evaluate closed-set accuracy / F1 and confusion matrix

def evaluate_set(X, y, name):
    pred = clf.predict(X)
    acc = accuracy_score(y, pred)
    f1  = f1_score(y, pred, average="macro")
    print(f"[{name}] Accuracy={acc:.4f}  Macro-F1={f1:.4f}")
    return acc, f1, pred

val_acc, val_f1, val_pred = evaluate_set(X_val, y_val, "Validation")
test_acc, test_f1, test_pred = evaluate_set(X_test, y_test, "Test")

# Confusion matrix on test set
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png", dpi=300)


# 6. Compute open-set threshold using cosine similarity

from sklearn.preprocessing import normalize
def class_centroids(X, y, n_classes):
    C = []
    for c in range(n_classes):
        v = X[y==c].mean(axis=0)
        v = np.asarray(v).ravel()
        v = v / (np.linalg.norm(v) + 1e-12)
        C.append(v)
    return np.vstack(C)

def max_cosine(X, C):
    Xn = normalize(X, norm="l2")
    sims = Xn @ C.T
    return np.asarray(sims.max(axis=1)).ravel()

Cmat = class_centroids(X_train, y_train, len(labels))
scores_known = max_cosine(X_val, Cmat)
scores_unk   = max_cosine(X_unk, Cmat)
y_ood = np.concatenate([np.ones_like(scores_known), np.zeros_like(scores_unk)])
scores_all = np.concatenate([scores_known, scores_unk])
fpr, tpr, th = roc_curve(y_ood, scores_all)
auroc = auc(fpr, tpr)
idx_eer = np.argmin(np.abs(fpr - (1-tpr)))
EER = (fpr[idx_eer] + (1-tpr[idx_eer]))/2
threshold = th[idx_eer]
print(f"[Open-set] AUROC={auroc:.4f}, EER={EER:.4f}, threshold={threshold:.3f}")


# 7. Save models and outputs

os.makedirs("models", exist_ok=True)
joblib.dump(vect, "models/tfidf_vectorizer.pkl")
joblib.dump(clf, "models/logistic_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
np.save("models/class_centroids.npy", Cmat)
with open("models/open_threshold.txt", "w") as f:
    f.write(str(threshold))
print("Models saved under /models")


pd.DataFrame({
    "metric":["val_acc","val_f1","test_acc","test_f1","open_auroc","open_eer","open_threshold"],
    "value":[val_acc,val_f1,test_acc,test_f1,auroc,EER,threshold]
}).to_csv("outputs/baseline_results.csv", index=False)
print("baseline_results.csv saved")


# 8. Save top-weighted features

feature_names = np.array(vect.get_feature_names_out())
coef = clf.coef_
top_words = []
for i, author in enumerate(labels):
    topn_idx = np.argsort(-coef[i])[:20]
    for j in topn_idx:
        top_words.append({"author": author, "ngram": feature_names[j], "weight": float(coef[i,j])})
pd.DataFrame(top_words).to_csv("outputs/top_weighted_features.csv", index=False, encoding="utf-8-sig")
print("top_weighted_features.csv saved")
