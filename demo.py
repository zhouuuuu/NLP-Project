import sys, os

# 将项目根目录加入模块搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))




from transformer.detector import OpenSetDetector
from rewrite.style_rewrite import style_rewrite
from transformer.inference import InferenceEngine
import streamlit as st
import torch

os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\huggingface_cache\hub"
os.environ["TRANSFORMERS_CACHE"] = r"D:\huggingface_cache\models"

# ✅ 这里改成绝对路径
CKPT_PATH = r"D:\nlpproject\transformer\checkpoints\microsoft-deberta-v3-large\authorship_model.pt"

engine = InferenceEngine(ckpt_path=CKPT_PATH)
detector = OpenSetDetector(ckpt_path=CKPT_PATH, tau_proto=0.45)

st.title("Author Style Detection Demo")

text = st.text_area("Enter text to analyze:")

if st.button("Predict"):
    if text.strip():
        result = engine.predict(text, top_k=5, return_embedding=True)
        author_pred = result["author"]
        prob_pred = result["prob"]

        label, score = detector.detect(text)
        if label == "Unknown":
            st.error(f"⚠️ 该文本可能来自未知作者（score={score:.3f}）")
        else:
            st.success(f"**Predicted Author:** {label}")
            st.write(f"**Confidence:** {prob_pred:.3f} | Open-set score={score:.3f}")

        st.bar_chart({item["author"]: item["prob"] for item in result["top_k"]})
    else:
        st.warning("Please enter some text first.")



st.markdown("---")
st.header("✍️ Style Rewrite System")

ckpt_path = r"D:\nlpproject\transformer\checkpoints\microsoft-deberta-v3-large\authorship_model.pt"

# 用户输入句子
rewrite_text = st.text_area("Enter text to rewrite:", key="rewrite_input")

# 从 checkpoint 读取作者列表
import torch
ck = torch.load(ckpt_path, map_location="cpu")
authors = ck["authors"]

target_author = st.selectbox("Select target author style:", authors)
strength = st.slider("Style strength", 0.5, 2.0, 1.0, 0.1)

if st.button("Rewrite Style"):
    if rewrite_text.strip():
        with st.spinner("Rewriting in progress..."):
            best, result = style_rewrite(
                src_text=rewrite_text,
                target_author=target_author,
                ckpt=ckpt_path,
                n_bt_each=3,  # 减少生成数避免太慢
                bridges=("de", "fr"),
                w_style=0.6, w_sem=0.35, w_ppl=0.05, strength=strength
            )

        st.subheader("✨ Best Rewrite Result")
        st.write(best)

        # 显示分数
        orig_t, sp, sv, ppl, sc = result["original"]
        st.markdown(f"**Original:** {orig_t}")
        st.markdown(f"Style={sp:.3f} | Sim={sv:.3f} | PPL={ppl:.1f} | Score={sc:.3f}")

        # 展示候选前 5 名
        st.markdown("**Top-5 Candidates:**")
        for t, sp, sv, ppl, sc in result["candidates"][:5]:
            st.markdown(f"- {t}")
            st.caption(f"Style={sp:.3f} | Sim={sv:.3f} | PPL={ppl:.1f} | Score={sc:.3f}")
    else:
        st.warning("Please enter a sentence first.")