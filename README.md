# NLP-Project

## Author-Style Recognition Baseline (TF-IDF + Logistic Regression)

This project implements a closed-set author classification and open-set author-style detection system using character n-gram TF-IDF features and a multinomial Logistic Regression baseline.

### Project Structure

```
author_style_system/
│
├── train_baseline.py          # Train model & export artifacts
├── predict_topk.py            # Inference interface
├── author_style_dataset_balanced.csv
│
├── models/                    # Saved model artifacts
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_model.pkl
│   ├── label_encoder.pkl
│   ├── class_centroids.npy
│   └── open_threshold.txt
│
└── outputs/                   # Evaluation results
    ├── baseline_results.csv
    ├── confusion_matrix.png
    └── top_weighted_features.csv
```

### 1. Training
Run the training script once:

```bash
python train_baseline.py
```

This will:

- Train a character n-gram TF-IDF + Logistic Regression model  
- Evaluate Accuracy / Macro-F1 / AUROC / EER  
- Compute class centroids for open-set detection  
- Save all artifacts under `/models` and results under `/outputs`

---

### 2. Inference (Top-k Prediction)

Use the inference script without retraining:

```bash
python predict_topk.py
```

Example output:

```
Enter a text snippet:
> The tone of this essay feels reflective yet slightly ironic.

Predicted author: Author_B | Unknown: False
Max cosine similarity: 0.721
Top-k similar authors:
  Author_B             -> 0.6032
  Author_D             -> 0.1983
  Author_A             -> 0.1237
  Author_F             -> 0.0618
  Author_C             -> 0.0123
```

---

### 3. Output Files

| File                                  | Description                                                  |
| ------------------------------------- | ------------------------------------------------------------ |
| **outputs/baseline_results.csv**      | Main evaluation metrics (Accuracy, Macro-F1, AUROC, etc.)    |
| **outputs/confusion_matrix.png**      | Confusion matrix of closed-set classification                |
| **outputs/top_weighted_features.csv** | Top 20 most informative n-grams per author (sorted by model weight) |
| **models/**                           | Folder storing the trained TF-IDF vectorizer, logistic model, label encoder, centroid vectors, and open-set threshold |

---

### Model Highlights

- Character-level TF-IDF (1–5-gram): captures writing style, punctuation, and lexical habits  
- Logistic Regression (multinomial): interpretable, fast, and strong baseline  
- Open-set detection: cosine similarity to class centroids with learned threshold  
- Top-k ranking: provides probability scores for similar authors  

---

###  Dependencies

- Python ≥ 3.8  
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `joblib`

Install them with:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

---

###  License

MIT License – for academic and research use.