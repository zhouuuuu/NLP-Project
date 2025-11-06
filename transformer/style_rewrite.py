#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style Rewrite System (fixed, final)
- Baseline (original sentence) is scored first and NOT included in candidate ranking
- Up to 10 top candidates are printed
- Uses custom AuthorModel (encoder -> proj -> classifier) to match your training
"""

import os, math, argparse, torch, numpy as np
from typing import List, Tuple, Dict, Any
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModel,                # encoder backbone (DeBERTa/RoBERTa/…)
    AutoModelForSeq2SeqLM,    # MT models for back-translation
    AutoModelForCausalLM,     # GPT-2 for perplexity
)

from sentence_transformers import SentenceTransformer, util as st_util

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_ONLY = os.getenv("TRANSFORMERS_OFFLINE", "0").lower() in ("1","true","yes")

# =============== 0. 自定义模型结构（与你训练时一致） ===============
class AuthorModel(nn.Module):
    """
    encoder(hidden H) -> proj(H -> H -> emb_dim) -> classifier(emb_dim -> C)
    use_mean_pool=True: masked mean pooling；False: 使用 CLS token。
    """
    def __init__(self, model_name: str, num_classes: int,
                 emb_dim: int = 256, use_mean_pool: bool = True,
                 local_only: bool = LOCAL_ONLY):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_only)
        self.use_mean_pool = use_mean_pool
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, emb_dim)
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, **batch):
        out = self.encoder(**batch)  # last_hidden_state: [B,T,H]
        if self.use_mean_pool:
            mask = batch["attention_mask"].unsqueeze(-1)  # [B,T,1]
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
        else:
            pooled = out.last_hidden_state[:, 0]           # CLS
        z = self.proj(pooled)                              # [B, emb_dim]
        logits = self.classifier(z)                        # [B, C]
        return logits, z

# =============== 1. 回译生成 ===============
class BackTranslator:
    def __init__(self, src="en", mid="de"):
        m1 = f"Helsinki-NLP/opus-mt-{src}-{mid}"
        m2 = f"Helsinki-NLP/opus-mt-{mid}-{src}"
        self.tok_en2mid = AutoTokenizer.from_pretrained(m1, local_files_only=LOCAL_ONLY)
        self.gen_en2mid = AutoModelForSeq2SeqLM.from_pretrained(m1, local_files_only=LOCAL_ONLY).to(DEVICE)
        self.tok_mid2en = AutoTokenizer.from_pretrained(m2, local_files_only=LOCAL_ONLY)
        self.gen_mid2en = AutoModelForSeq2SeqLM.from_pretrained(m2, local_files_only=LOCAL_ONLY).to(DEVICE)

    @torch.no_grad()
    def paraphrase(self, text: str, num_samples=4, max_len=256, top_k=50, top_p=0.95, temperature=1.0) -> List[str]:
        enc = self.tok_en2mid([text], return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
        mid = self.gen_en2mid.generate(**enc, do_sample=True, num_return_sequences=num_samples,
                                       top_k=top_k, top_p=top_p, temperature=temperature, max_length=max_len,num_beams=1)
        mids = self.tok_en2mid.batch_decode(mid, skip_special_tokens=True)
        batch = self.tok_mid2en(mids, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEVICE)
        out = self.gen_mid2en.generate(**batch, do_sample=True, num_return_sequences=1, max_length=max_len, num_beams=1)
        paras = self.tok_mid2en.batch_decode(out, skip_special_tokens=True)
        # 去重 + 去空 + 去与原文完全相同的句子
        uniq = []
        for p in paras:
            p = p.strip()
            if p and p != text and p not in uniq:
                uniq.append(p)
        return uniq

# =============== 2. 风格打分（加载你训练好的 ckpt） ===============
class StyleScorer:
    def __init__(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" not in ck or "authors" not in ck:
            raise KeyError("Checkpoint must contain 'state_dict' and 'authors'.")

        self.authors = ck["authors"]
        model_name = ck.get("model_name", "microsoft/deberta-v3-large")
        use_mean_pool = ck.get("use_mean_pool", True)
        emb_dim = ck["state_dict"]["classifier.weight"].shape[1]

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False, local_files_only=LOCAL_ONLY)
        self.model = AuthorModel(model_name,
                                 num_classes=len(self.authors),
                                 emb_dim=emb_dim,
                                 use_mean_pool=use_mean_pool,
                                 local_only=LOCAL_ONLY).to(DEVICE)
        missing, unexpected = self.model.load_state_dict(ck["state_dict"], strict=False)
        if missing or unexpected:
            print(f"[Warn] load_state_dict mismatch. missing={missing}, unexpected={unexpected}")
        self.model.eval()

    @torch.no_grad()
    def prob(self, text: str, author_name: str) -> float:
        if author_name not in self.authors:
            raise ValueError(f"Author '{author_name}' not in ckpt authors: {self.authors}")
        idx = self.authors.index(author_name)
        enc = self.tok(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits, _ = self.model(**enc)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return float(probs[idx])

# =============== 3. 语义相似度 + 流畅度 ===============
class SimFluency:
    def __init__(self):
        self.emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
        self.gpt_tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=LOCAL_ONLY)
        self.gpt = AutoModelForCausalLM.from_pretrained("gpt2", local_files_only=LOCAL_ONLY).to(DEVICE).eval()

    @torch.no_grad()
    def sim(self, a: str, b: str) -> float:
        ea = self.emb.encode([a], convert_to_tensor=True, normalize_embeddings=True)
        eb = self.emb.encode([b], convert_to_tensor=True, normalize_embeddings=True)
        return float(st_util.cos_sim(ea, eb).item())

    @torch.no_grad()
    def ppl(self, text: str, max_len=512) -> float:
        ids = self.gpt_tok.encode(text, return_tensors="pt", truncation=True, max_length=max_len).to(DEVICE)
        out = self.gpt(input_ids=ids, labels=ids)
        loss = out.loss
        return float(math.exp(loss.item()))

# =============== 4. 核心流程 ===============
def style_rewrite(
    src_text: str, target_author: str, ckpt: str,
    n_bt_each=10, bridges=("de", "fr", "it"),
    w_style=0.6, w_sem=0.35, w_ppl=0.05, strength=1.0
) -> Tuple[str, Dict[str, Any]]:
    """
    返回:
      best_text, {
        'original': (text, p_style, sim, ppl, score),
        'candidates': [(text, p_style, sim, ppl, score), ...]  # 排序后
      }
    """
    scorer = StyleScorer(ckpt)
    util = SimFluency()

    # 1) baseline: 原句评分（不参与排序）
    orig_style = scorer.prob(src_text, target_author)
    orig_sim = 1.0
    orig_ppl = util.ppl(src_text)
    orig_score = (w_style * orig_style * strength) + (w_sem * orig_sim) + (w_ppl * 1.0)
    original = (src_text, float(orig_style), float(orig_sim), float(orig_ppl), float(orig_score))

    # 2) 生成候选
    cset = set()
    for mid in bridges:
        bt = BackTranslator("en", mid)
        for c in bt.paraphrase(src_text, num_samples=n_bt_each):
            cset.add(c)
    # 不让原句参与候选排序
    if src_text in cset:
        cset.remove(src_text)

    cands = list(cset)
    if not cands:
        return src_text, {"original": original, "candidates": []}

    # 3) 候选打分
    style_probs, sims, ppl_vals = [], [], []
    for t in cands:
        style_probs.append(scorer.prob(t, target_author))
        sims.append(util.sim(src_text, t))
        ppl_vals.append(util.ppl(t))

    ppl_arr = np.array(ppl_vals, dtype=float)
    ppl_norm = 1.0 - (ppl_arr - ppl_arr.min()) / (np.ptp(ppl_arr) + 1e-8)  # NumPy 2.0 适配

    scored = []
    for t, sp, sv, pn, raw_ppl in zip(cands, style_probs, sims, ppl_norm, ppl_vals):
        score = (w_style * sp * strength) + (w_sem * sv) + (w_ppl * pn)
        scored.append((t, float(sp), float(sv), float(raw_ppl), float(score)))

    best = max(scored, key=lambda x: x[-1])
    scored_sorted = sorted(scored, key=lambda x: x[-1], reverse=True)
    return best[0], {"original": original, "candidates": scored_sorted}

# =============== 5. CLI & 输出 ===============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="e.g. checkpoints/microsoft-deberta-v3-large/authorship_model.pt")
    ap.add_argument("--author", required=True, help="Target author name (must be in ckpt['authors'])")
    ap.add_argument("--text", required=True, help="Input text to rewrite")
    ap.add_argument("--strength", type=float, default=1.0, help="Style strength multiplier (>1 favors style)")
    ap.add_argument("--out", type=str, default="rewrite_result.txt")
    args = ap.parse_args()

    best, result = style_rewrite(
        src_text=args.text,
        target_author=args.author,
        ckpt=args.ckpt,
        n_bt_each=10, bridges=("de","fr","it"),
        w_style=0.6, w_sem=0.35, w_ppl=0.05, strength=args.strength
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("=== Source ===\n" + args.text.strip() + "\n\n")

        # Baseline
        t, sp, sv, ppl, sc = result["original"]
        f.write("=== Original Sentence (Baseline) ===\n")
        f.write(f"{t}\n  -> p_style={sp:.3f} | sim={sv:.3f} | ppl={ppl:.1f} | score={sc:.3f}\n\n")

        # Best
        f.write(f"=== Best ({args.author}) ===\n{best}\n\n")

        # Top-10 candidates
        f.write("=== Candidates (text | p_style | sim | ppl | score) ===\n")
        for t, sp, sv, ppl, sc in result["candidates"][:10]:
            f.write(f"{t}\n  -> {sp:.3f} | {sv:.3f} | {ppl:.1f} | {sc:.3f}\n\n")

    print(f"[DONE] Saved to {args.out}\n[BEST] {best}")

if __name__ == "__main__":
    main()
