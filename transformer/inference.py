import os
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn

# -------- Config (可以用环境变量覆盖) --------
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/deberta-v3-large")
MODEL_TAG  = MODEL_NAME.replace("/", "-")
DEFAULT_CKPT = f"./checkpoints/{MODEL_TAG}/authorship_model.pt"
CKPT_PATH = os.getenv("CKPT_PATH", DEFAULT_CKPT)
MAX_LEN = int(os.getenv("MAX_LEN", 512))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_ONLY = os.getenv("TRANSFORMERS_OFFLINE", "0") not in ("0", "", "false", "False")

class AuthorModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int, emb_dim: int = 256, use_mean_pool: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=LOCAL_ONLY)
        self.use_mean_pool = use_mean_pool
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, emb_dim))
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, **batch):
        out = self.encoder(**batch)
        if self.use_mean_pool:
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        else:
            pooled = out.last_hidden_state[:, 0]
        z = self.proj(pooled)
        logits = self.classifier(z)
        return logits, z

class InferenceEngine:
    def __init__(self, ckpt_path: str = CKPT_PATH, model_name: str = MODEL_NAME, max_len: int = MAX_LEN, device: str = DEVICE):
        self.device = device
        self.max_len = max_len

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\n"
                                    f"Hint: expected at ./checkpoints/{MODEL_TAG}/authorship_model.pt")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.authors = ckpt["authors"]

        # 优先使用 ckpt 里的真实模型名 & pooling 设定
        model_name = ckpt.get("model_name", model_name)
        use_mean_pool = ckpt.get("use_mean_pool", True)
        # 从权重反推 emb_dim，避免不一致
        emb_dim = ckpt["state_dict"]["classifier.weight"].shape[1]

        self.model = AuthorModel(model_name,
                                 num_classes=len(self.authors),
                                 emb_dim=emb_dim,
                                 use_mean_pool=use_mean_pool)
        missing, unexpected = self.model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing or unexpected:
            print(f"[Warn] load_state_dict mismatch. missing={missing}, unexpected={unexpected}")

        self.model.to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=LOCAL_ONLY)

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits, z = self.model(**enc)
        return logits, z

    @staticmethod
    def _softmax(x: torch.Tensor) -> np.ndarray:
        return torch.softmax(x, dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict(self, text: str, top_k: int = 5, return_embedding: bool = False) -> Dict[str, Any]:
        logits, z = self._encode([text])
        probs = self._softmax(logits)[0]
        top_idx = np.argsort(probs)[::-1][:top_k]
        top = [{"author": self.authors[i], "prob": float(probs[i])} for i in top_idx]
        result = {
            "author": top[0]["author"],
            "prob": top[0]["prob"],
            "top_k": top,
            "logits": logits.squeeze(0).cpu().numpy().tolist()
        }
        if return_embedding:
            result["embedding"] = z.squeeze(0).cpu().numpy().tolist()
        return result

    @torch.no_grad()
    def predict_batch(self, texts: List[str], top_k: int = 5, return_embedding: bool = False) -> List[Dict[str, Any]]:
        logits, z = self._encode(texts)
        probs = self._softmax(logits)
        out = []
        for i in range(len(texts)):
            p = probs[i]
            top_idx = np.argsort(p)[::-1][:top_k]
            top = [{"author": self.authors[j], "prob": float(p[j])} for j in top_idx]
            item = {
                "author": top[0]["author"],
                "prob": top[0]["prob"],
                "top_k": top,
                "logits": logits[i].cpu().numpy().tolist()
            }
            if return_embedding:
                item["embedding"] = z[i].cpu().numpy().tolist()
            out.append(item)
        return out

if __name__ == "__main__":
    engine = InferenceEngine()
    demo = "This is a short passage to test author-style classification."
    print(engine.predict(demo, top_k=5, return_embedding=False))
