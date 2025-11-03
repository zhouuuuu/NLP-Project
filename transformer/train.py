"""
train_transformer.py
--------------------
多模型兼容版 Transformer 文风分类训练脚本
支持 DeBERTa / RoBERTa / BERT / MPNet 等。
"""

import os
import math
import time
import json
import random
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
import matplotlib.pyplot as plt

# =====================================================
# 🔧 环境参数（可被 Slurm 阵列脚本自动注入）
# =====================================================
DATA_PATH = os.getenv("DATA_PATH", "author_style_dataset_balanced.csv")
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/deberta-v3-large")

SAVE_DIR = os.getenv("SAVE_DIR", f"./checkpoints/{MODEL_NAME.replace('/', '-')}")
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = int(os.getenv("EPOCHS", 8))
LR = float(os.getenv("LR", 2e-5))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
GRAD_ACCUM_STEPS = int(os.getenv("GRAD_ACCUM_STEPS", 1))
SUPCON_LAMBDA = float(os.getenv("SUPCON_LAMBDA", 0.1))
SUPCON_TAU = float(os.getenv("SUPCON_TAU", 0.07))
MAX_LEN = int(os.getenv("MAX_LEN", 512))
USE_MEAN_POOL = os.getenv("USE_MEAN_POOL", "True").lower() == "true"
USE_AMP = os.getenv("USE_AMP", "True").lower() == "true"
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", 0.1))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
SEED = int(os.getenv("SEED", 42))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 🧱 固定随机种子
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

# =====================================================
# 📚 Dataset
# =====================================================
class AuthorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, author2id: Dict[str, int]):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.author2id = author2id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.author2id[row["author"]], dtype=torch.long)
        return item

# =====================================================
# 🧠 Model
# =====================================================
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
        labels = batch.pop("labels", None)
        out = self.encoder(**batch)
        if self.use_mean_pool:
            last_hidden = out.last_hidden_state
            attn_mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * attn_mask).sum(1) / attn_mask.sum(1).clamp_min(1e-6)
        else:
            pooled = out.last_hidden_state[:, 0]
        z = self.proj(pooled)
        logits = self.classifier(z)
        return logits, z, labels

# =====================================================
# 🔥 SupCon Loss（稳定版）
# =====================================================
def supcon_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07):
    z = nn.functional.normalize(z, dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    labels = y.view(-1, 1)
    mask_pos = (labels == labels.T).float()
    mask_pos.fill_diagonal_(0)
    logits = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_counts = mask_pos.sum(1)
    mean_log_pos = (mask_pos * log_prob).sum(1) / pos_counts.clamp_min(1.0)
    valid = (pos_counts > 0).float()
    loss = -(valid * mean_log_pos).sum() / valid.sum().clamp_min(1.0)
    return loss

# =====================================================
# 🧪 Evaluate
# =====================================================
@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, authors: List[str]):
    model.eval()
    preds, gts = [], []
    for batch in dataloader:
        y = batch["labels"].cpu().numpy().tolist()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits, _, _ = model(**batch)
        p = logits.argmax(-1).cpu().numpy().tolist()
        preds += p
        gts += y
    f1 = f1_score(gts, preds, average="macro", zero_division=0)
    return f1, gts, preds

# =====================================================
# 🏋️ Train
# =====================================================
def train_epoch(model, dataloader, optimizer, scheduler, criterion, scaler, epoch_idx):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(dataloader, desc=f"Training (epoch {epoch_idx})"), start=1):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            logits, z, y = model(**batch)
            ce = criterion(logits, y)
            loss = ce + (SUPCON_LAMBDA * supcon_loss(z, y, SUPCON_TAU) if SUPCON_LAMBDA > 0 else 0.0)
            loss = loss / GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()

        if step % GRAD_ACCUM_STEPS == 0 or step == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
    return total_loss / max(1, len(dataloader))

# =====================================================
# 🚀 Main
# =====================================================
def main():
    print(f"[Info] Model: {MODEL_NAME} | Save Dir: {SAVE_DIR}")
    df = pd.read_csv(DATA_PATH)
    df["split"] = df["split"].astype(str).str.lower().replace({
        "valid": "val", "validation": "val", "dev": "val"
    })
    authors = sorted(df["author"].unique())
    author2id = {a: i for i, a in enumerate(authors)}
    print(f"Loaded {len(df)} samples | authors={len(authors)}")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    train_loader = DataLoader(
        AuthorDataset(train_df, tokenizer, author2id),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        AuthorDataset(val_df, tokenizer, author2id),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = AuthorModel(MODEL_NAME, num_classes=len(authors), use_mean_pool=USE_MEAN_POOL).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_f1 = 0.0
    train_losses, val_f1s = [], []

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch)
        val_f1, y_true, y_pred = evaluate(model, val_loader, authors)
        print(f"[Epoch {epoch}] loss={loss:.4f}, val_F1={val_f1:.4f}")

        train_losses.append(loss)
        val_f1s.append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "state_dict": model.state_dict(),
                "authors": authors,
                "model_name": MODEL_NAME,
                "use_mean_pool": USE_MEAN_POOL
            }, os.path.join(SAVE_DIR, "authorship_model.pt"))
            with open(os.path.join(SAVE_DIR, "results.jsonl"), "a") as f:
                f.write(json.dumps({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": MODEL_NAME,
                    "val_macro_f1": float(val_f1),
                    "supcon_lambda": SUPCON_LAMBDA,
                    "lr": LR,
                    "epochs": EPOCHS
                }) + "\n")
            print(f"✅ Saved new best model to {SAVE_DIR} (F1={val_f1:.4f})")

    # 曲线
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_f1s)+1), val_f1s, label="Val Macro-F1")
    plt.xlabel("Epoch"); plt.legend(); plt.title(f"{MODEL_NAME} Training")
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"))
    print(f"Saved training curve to {SAVE_DIR}/training_curve.png")

    # Final eval
    best_ckpt = torch.load(os.path.join(SAVE_DIR, "authorship_model.pt"), map_location="cpu")
    model.load_state_dict(best_ckpt["state_dict"])
    model.to(DEVICE).eval()

    print("\n=== Final Validation ===")
    val_f1, y_true, y_pred = evaluate(model, val_loader, authors)
    val_report = classification_report(y_true, y_pred, target_names=authors, digits=3)
    val_matrix = confusion_matrix(y_true, y_pred)
    print(val_report)
    print(val_matrix)

    test_f1 = None
    ty, tp = [], []

    if len(test_df) > 0:
        test_loader = DataLoader(
            AuthorDataset(test_df, tokenizer, author2id),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        test_f1, ty, tp = evaluate(model, test_loader, authors)
        test_report = classification_report(ty, tp, target_names=authors, digits=3)
        test_matrix = confusion_matrix(ty, tp)

        print("\n=== Test Set ===")
        print(f"Macro-F1={test_f1:.4f}")
        print(test_report)
        print(test_matrix)
    else:
        test_report, test_matrix = "No test split found.\n", np.array([])

    # =========================================================
    # ✅ Save evaluation results
    # =========================================================
    report_path = os.path.join(SAVE_DIR, "eval_report.txt")
    with open(report_path, "w") as f:
        f.write(f"=== Validation Results ===\n")
        f.write(f"Macro-F1={val_f1:.4f}\n\n")
        f.write(val_report + "\n")
        f.write(np.array2string(val_matrix))
        f.write("\n\n=== Test Results ===\n")
        if test_f1 is not None:
            f.write(f"Macro-F1={test_f1:.4f}\n\n")
        f.write(test_report)
        f.write("\n")
        f.write(np.array2string(test_matrix))

    np.save(os.path.join(SAVE_DIR, "val_confmat.npy"), val_matrix)
    if len(test_df) > 0:
        np.save(os.path.join(SAVE_DIR, "test_confmat.npy"), test_matrix)

    print(f"\n✅ Saved full evaluation report to {report_path}")
    print(f"✅ Saved confusion matrices to {SAVE_DIR}/val_confmat.npy "
          f"and {SAVE_DIR}/test_confmat.npy (if test split exists)")

if __name__ == "__main__":
    main()
