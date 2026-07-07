import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Figure 2: Noise Robustness Analysis (Single vs Proposed)
# x-axis: WER (Noise Level), y-axis: Accuracy
# + Horizontal dashed baselines at WER=0.0 for each method
# =========================================================

# 1) x-axis (WER points)
wer = np.array([0.00, 0.10, 0.20, 0.30, 0.40])

# 2) Noise ordering (Distractive Content -> Shuffling)
noise_order = [
    "Typographical Error",
    "Word Deletion",
    "Word Duplication",
    "OCR Error",
    "Shuffling",
    "Word Substitution",
]

# 3) Data (예시)
# - Single은 네가 준 값 그대로 유지
# - Proposed 값은 네가 가지고 있는 값으로 채워 넣으면 됨 (지금은 예시/플레이스홀더)
base_acc_single = 0.6622

data = {
    "Typographical Error": {
        "Single":   [0.6622, 0.6300, 0.6118, 0.5973, 0.5739],
        "Proposed": [0.7808, 0.7606, 0.7518, 0.7491, 0.7423],
    },
    "Word Deletion": {
        "Single":   [0.6622, 0.6372, 0.6168, 0.5972, 0.5634],
        "Proposed": [0.7808, 0.7511, 0.7342, 0.7188, 0.7001],
    },
    "Word Duplication": {
        "Single":   [0.6622, 0.6545, 0.6462, 0.6451, 0.6402],
        "Proposed": [0.7808, 0.7665, 0.7670, 0.7654, 0.7672],
    },
    "OCR Error": {
        "Single":   [0.6622, 0.6346, 0.6182, 0.5973, 0.5683],
        "Proposed": [0.7808, 0.7625, 0.7555, 0.7559, 0.7519],
    },
    "Shuffling": {
        "Single":   [0.6622, 0.6437, 0.6273, 0.6160, 0.6096],
        "Proposed": [0.7808, None, None, None, None],
    },
    "Word Substitution": {
        "Single":   [0.6622, 0.6130, 0.5712, 0.5302, 0.5034],
        "Proposed": [0.7808, None, None, None, None],
    },
}


# 4) Plot settings (이미지 스타일에 가깝게)
fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
axes = axes.flatten()

for ax, noise in zip(axes, noise_order):
    y_single = np.array(data[noise]["Single"], dtype=float)
    y_prop   = np.array(data[noise]["Proposed"], dtype=float)

    # 실선 2개
    line_single, = ax.plot(
        wer, y_single, marker="o", markersize=4, linewidth=1.6,
        label="Single LLM"
    )
    line_prop, = ax.plot(
        wer, y_prop, marker="o", markersize=4, linewidth=1.6,
        label="Proposed"
    )

    # ---- 가로 점선(---): WER=0.0 기준으로 x축 끝까지 ----
    # Single 기준선
    ax.axhline(
        y=y_single[0],
        linestyle="--",
        linewidth=1.2,
        color=line_single.get_color(),
        alpha=0.7
    )
    # Proposed 기준선
    ax.axhline(
        y=y_prop[0],
        linestyle="--",
        linewidth=1.2,
        color=line_prop.get_color(),
        alpha=0.7
    )

    ax.set_title(noise, fontsize=13, pad=10)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.0, 0.4)
    ax.set_xticks(wer)

# 이미지처럼 넓은 y범위
for ax in axes:
    ax.set_ylim(0.35, 0.92)

# 공통 라벨
fig.supylabel("Accuracy", fontsize=14, x=0.04)
for ax in axes[3:]:  # 하단 row
    ax.set_xlabel("WER", fontsize=13)

# 공통 범례 (하단 중앙)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
    fontsize=12
)

plt.tight_layout(rect=[0.03, 0.05, 0.995, 0.98])
plt.savefig("figure2_like_image_with_shuffling.png", dpi=300, bbox_inches="tight")
plt.show()
