import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Figure 2 template: 6 noise types, 2x3 subplots
# x-axis: WER, y-axis: Accuracy
# Lines: Single LLM vs Proposed (multi-agent)
#
# You will later replace the dummy y-values in `data`.
# =========================================================

# 1) x-axis (WER points) - 마음대로 늘리거나 바꿔도 됨
wer = np.array([0.00, 0.10, 0.20, 0.30, 0.40])

# 2) Noise ordering (subplot 순서)
noise_order = [
    "Typographical Error",
    "Word Deletion",
    "Word Duplication",
    "OCR Error",
    "Distractive Content",
    "Word Substitution",
]

# 3) Dummy data (여기만 나중에 네 실험 결과로 교체)
#    각 noise마다: {"Single": [..], "Proposed": [..]}
#    길이는 wer와 동일해야 함.
data = {
    "Typographical Error": {
        "Single":   [0.88, 0.85, 0.80, 0.73, 0.62],
        "Proposed": [0.90, 0.88, 0.85, 0.81, 0.77],
    },
    "Word Deletion": {
        "Single":   [0.88, 0.83, 0.77, 0.68, 0.55],
        "Proposed": [0.90, 0.87, 0.83, 0.78, 0.73],
    },
    "Word Duplication": {
        "Single":   [0.88, 0.84, 0.79, 0.71, 0.60],
        "Proposed": [0.90, 0.88, 0.84, 0.80, 0.76],
    },
    "OCR Error": {
        "Single":   [0.88, 0.82, 0.74, 0.63, 0.48],
        "Proposed": [0.90, 0.86, 0.81, 0.74, 0.68],
    },
    "Distractive Content": {
        "Single":   [0.88, 0.80, 0.70, 0.55, 0.38],
        "Proposed": [0.90, 0.86, 0.80, 0.73, 0.66],
    },
    "Word Substitution": {
        "Single":   [0.88, 0.86, 0.83, 0.79, 0.74],
        "Proposed": [0.90, 0.89, 0.87, 0.85, 0.82],
    },
}

# 4) Plot settings
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

for ax, noise in zip(axes, noise_order):
    y_single = np.array(data[noise]["Single"], dtype=float)
    y_prop   = np.array(data[noise]["Proposed"], dtype=float)

    # Basic sanity checks (optional)
    if len(y_single) != len(wer) or len(y_prop) != len(wer):
        raise ValueError(f"Length mismatch in '{noise}': "
                         f"WER has {len(wer)} points, but Single/Proposed differ.")

    ax.plot(wer, y_single, marker="o", markersize=3, linewidth=1, label="Single LLM")
    ax.plot(wer, y_prop,   marker="o", markersize=3, linewidth=1, label="Proposed")

    ax.set_title(noise, fontsize=10, pad=6)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, 1.0)

# 공통 x/y 라벨
fig.supylabel("Accuracy", fontsize=12, x=0.04)
for ax in axes[3:]:   # 하단 row (index 3,4,5)
    ax.set_xlabel("WER", fontsize=11)

for ax in axes:
    ax.set_ylim(0.35, 0.92)
    
# legend 위치를 더 아래로
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 0.01)
)

# 레이아웃 여유 공간 확보
plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.96])

# 5) Save (optional)
plt.savefig("figure2_dummy.png", dpi=300, bbox_inches="tight")
#plt.savefig("figure2_dummy.pdf", bbox_inches="tight")

plt.show()
