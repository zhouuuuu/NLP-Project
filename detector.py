#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Open-set detection module (based on D's open_set_eval.ipynb)
-------------------------------------------------------------
Implements:
- Max Probability
- Energy Score
- Prototype Distance
for author identification tasks.

Usage:
    from detector import OpenSetDetector
    detector = OpenSetDetector(
        ckpt_path="checkpoints/authorship_model.pt",
        tau_proto=0.45
    )
    label, score = detector.detect("Some text here")
"""

import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.special import logsumexp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 1. AuthorModel (aligns with train.py / inference.py)
# ------------------------------------------------------------------
class AuthorModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, emb_dim: int = 256, use_mean_pool: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.use_mean_pool = use_mean_pool
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, **batch):
        out = self.encoder(**{k:v for k,v in batch.items() if k in ("input_ids","attention_mask")})
        if self.use_mean_pool:
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        else:
            pooled = out.last_hidden_state[:, 0]
        z = self.proj(pooled)
        logits = self.classifier(z)
        return logits, z


# ------------------------------------------------------------------
# 2. OpenSetDetector
# ------------------------------------------------------------------
class OpenSetDetector:
    def __init__(self, ckpt_path: str, tau_proto: float = 0.45, tau_energy: float = 0.0, metric: str = "euclidean"):
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        ck = torch.load(ckpt_path, map_location="cpu")

        self.authors = ck["authors"]
        model_name = ck.get("model_name", "microsoft/deberta-v3-large")
        use_mean_pool = ck.get("use_mean_pool", True)
        emb_dim = ck["state_dict"]["classifier.weight"].shape[1]

        self.model = AuthorModel(model_name, num_classes=len(self.authors), emb_dim=emb_dim, use_mean_pool=use_mean_pool).to(DEVICE)
        self.model.load_state_dict(ck["state_dict"], strict=False)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        self.metric = metric
        self.tau_proto = tau_proto
        self.tau_energy = tau_energy
        self.centroids = None

    # ---------------------- helpers ----------------------
    def _softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def _energy_score(self, logits, T=1.0):
        return -(-T * logsumexp(logits / T, axis=1))  # 越高越“已知”

    def _compute_centroids(self, embeddings, labels):
        centroids = {}
        for i, a in enumerate(self.authors):
            mask = labels == i
            if mask.sum() == 0:
                centroids[i] = np.zeros(embeddings.shape[1])
            else:
                centroids[i] = embeddings[mask].mean(axis=0)
        self.centroids = centroids

    def _proto_score(self, emb):
        assert self.centroids is not None, "Centroids not initialized!"
        centroid_matrix = np.stack([v for v in self.centroids.values()], axis=0)
        if self.metric == "euclidean":
            d = euclidean_distances(emb, centroid_matrix)
        else:
            d = cosine_distances(emb, centroid_matrix)
        min_d = d.min(axis=1)
        return -min_d  # 越高越“已知”

    # ---------------------- public API ----------------------
    @torch.no_grad()
    def extract_features(self, text: str):
        batch = self.tokenizer([text], truncation=True, padding=True, return_tensors="pt", max_length=512).to(DEVICE)
        logits, z = self.model(**batch)
        return logits.cpu().numpy(), z.cpu().numpy()

    @torch.no_grad()
    def detect(self, text: str):
        logits, z = self.extract_features(text)
        probs = self._softmax(logits)
        pred_label = int(np.argmax(probs))
        energy = self._energy_score(logits)[0]

        if self.centroids is not None:
            proto_score = self._proto_score(z)[0]
            if proto_score < self.tau_proto:
                return "Unknown", float(proto_score)
            else:
                return self.authors[pred_label], float(proto_score)
        else:
            if energy < self.tau_energy:
                return "Unknown", float(energy)
            else:
                return self.authors[pred_label], float(energy)

