# cm_plot.py
import numpy as np
import matplotlib.pyplot as plt

# 你的 9×9 混淆矩阵（来自测试结果）
cm = np.array([
    [72, 0, 0, 6, 0, 0, 0, 15, 18],
    [0, 76, 0, 1, 0, 0, 1, 2, 0],
    [49, 6, 89, 1, 0, 0, 2, 13, 32],
    [0, 11, 0, 146, 5, 0, 1, 9, 0],
    [0, 23, 0, 84, 191, 1, 1, 1, 0],
    [0, 1, 0, 1, 0, 166, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 38, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 101, 0],
    [0, 4, 0, 0, 0, 0, 2, 0, 287]
], dtype=float)

labels = [
    "ArthurConanDoyle", "CharlesDickens", "Chesterton",
    "EdgarAllanPoe", "HermanMelville", "JaneAusten",
    "MarkTwain", "OscarWilde", "VirginiaWoolf"
]

def plot_confusion_matrix(cm, labels, normalize=True, title="Confusion Matrix (Normalized)", save_path="confusion_matrix_heatmap.png"):
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # 处理某些行全 0 的情况

    plt.figure(figsize=(8.5, 7))
    im = plt.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    plt.title(title, fontsize=14)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Proportion" if normalize else "Count")

    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=60, ha="right", fontsize=8)
    plt.yticks(ticks, labels, fontsize=8)
    plt.xlabel("Predicted Author", fontsize=10)
    plt.ylabel("True Author", fontsize=10)

    # 在格子里标数值
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            disp = f"{val:.2f}" if normalize else f"{int(val)}"
            # 深色格子用白字，浅色格子用黑字
            color = "white" if (normalize and val > 0.6) else "black"
            plt.text(j, i, disp, ha="center", va="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved figure to: {save_path}")

if __name__ == "__main__":
    plot_confusion_matrix(cm, labels, normalize=True)
