"""Generate the Bi-NAC evaluation-results figures for the slide deck.

All numbers are taken from Singh et al., "RL with Learnable Textual Feedback:
A Bilevel Approach" (ICML 2026): Fig. 3, Fig. 4, Tables 1-5.
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 13,
    "axes.edgecolor": "#C9C8C2",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#ECEBE6",
    "grid.linewidth": 1,
    "axes.axisbelow": True,
})

BLUE = "#2a78d6"     # Bi-NAC
GRAY = "#B4B3AD"     # baselines
DARK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#8A9099"


def panel_title(ax, letter, title, sub=None):
    ax.set_title("", loc="left")
    ax.text(0, 1.13, f"{letter}  {title}", transform=ax.transAxes, fontsize=16,
            fontweight="bold", color=DARK, ha="left", va="bottom")
    if sub:
        ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=12, color=MUTED,
                ha="left", va="bottom")


def legend_patch(ax, items, loc="lower right", ncol=1):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for l, c in items], loc=loc,
              ncol=ncol, frameon=False, fontsize=12)


# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=200)
fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.12, hspace=0.62, wspace=0.48)
fig.text(0.04, 0.95, "Evaluation results (1/2): does Bi-NAC beat the state of the art?",
         fontsize=24, fontweight="bold", color=DARK)

# (a) Bi-NAC vs best baseline (SCoRe) across scales - dumbbell
ax = axes[0, 0]
rows = ["MATH · 1B", "MBPP · 1B", "GPQA · 1B",
        "MATH · 3B", "MBPP · 3B", "GPQA · 3B",
        "MATH · 8B", "MBPP · 8B", "GPQA · 8B"]
score = [39.8, 62.3, 34.4, 45.2, 70.7, 42.6, 56.8, 77.3, 44.6]
binac = [46.6, 66.4, 40.6, 51.4, 75.0, 49.3, 60.2, 79.8, 56.3]
ys = list(range(len(rows)))[::-1]
for y, s, b in zip(ys, score, binac):
    ax.plot([s, b], [y, y], color="#D9D8D2", lw=3, zorder=1, solid_capstyle="round")
    ax.text(b + 1.2, y, f"+{b - s:.1f}", va="center", fontsize=12, color=DARK,
            fontweight="bold" if b - s > 10 else "normal")
ax.scatter(score, ys, s=70, color=GRAY, zorder=2, label="SCoRe (best baseline)",
           edgecolor="white", linewidth=1.5)
ax.scatter(binac, ys, s=70, color=BLUE, zorder=3, label="Bi-NAC",
           edgecolor="white", linewidth=1.5)
ax.set_yticks(ys, rows)
ax.set_xlim(30, 88)
ax.set_xlabel("Accuracy (%)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=12,
          handletextpad=0.2, columnspacing=1.0)
panel_title(ax, "A", "Wins at every scale and benchmark",
            "LLaMA-3.2 / 3.1 (Fig. 3)")

# (b) Feedback baselines on MATH-500 1B
ax = axes[0, 1]
names = ["ArCHer  (RL actor, no critic)", "Dr. GRPO  (auxiliary critic)",
         "Critique-GRPO  (auxiliary critic)", "SCoRe  (fixed feedback)", "Bi-NAC  (learned feedback)"]
vals = [30.6, 38.0, 39.6, 39.8, 46.6]
cols = [GRAY] * 4 + [BLUE]
bars = ax.barh(range(len(vals)), vals, color=cols, height=0.62)
for i, v in enumerate(vals):
    ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=12, color=DARK,
            fontweight="bold" if i == 4 else "normal")
ax.set_yticks(range(len(vals)), names)
ax.set_xlim(0, 55)
ax.set_xlabel("MATH-500 accuracy (%)")
ax.grid(axis="y", visible=False)
panel_title(ax, "B", "Fixed / auxiliary critics plateau at ~38–40%",
            "MATH-500, 1B scale (Fig. 4, left)")

# (c) Parameter efficiency - smaller Bi-NAC vs larger GRPO
ax = axes[1, 0]
bench = ["MATH", "MBPP", "GPQA"]
grpo3, bin2 = [41.43, 61.55, 36.43], [46.56, 66.73, 41.20]
grpo7, bin6 = [48.40, 72.20, 43.60], [51.40, 75.00, 49.30]
w = 0.36
x1 = [0, 1, 2]
x2 = [3.6, 4.6, 5.6]
ax.bar([x - w / 2 - 0.01 for x in x1], grpo3, w, color=GRAY)
ax.bar([x + w / 2 + 0.01 for x in x1], bin2, w, color=BLUE)
ax.bar([x - w / 2 - 0.01 for x in x2], grpo7, w, color=GRAY)
ax.bar([x + w / 2 + 0.01 for x in x2], bin6, w, color=BLUE)
for xs, g, b in [(x1, grpo3, bin2), (x2, grpo7, bin6)]:
    for x, gv, bv in zip(xs, g, b):
        ax.text(x + w / 2, bv + 1, f"+{bv - gv:.1f}", ha="center", fontsize=11.5, color=DARK)
ax.set_xticks(x1 + x2, bench * 2)
ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy (%)")
ax.grid(axis="x", visible=False)
ax.text(1, -0.2, "Bi-NAC 2B  vs.  GRPO 3B", transform=ax.get_xaxis_transform(),
        ha="center", fontsize=12.5, fontweight="bold", color=MUTED)
ax.text(4.6, -0.2, "Bi-NAC 6B  vs.  GRPO 7B", transform=ax.get_xaxis_transform(),
        ha="center", fontsize=12.5, fontweight="bold", color=MUTED)
legend_patch(ax, [("GRPO (larger)", GRAY), ("Bi-NAC (smaller)", BLUE)], loc="upper left", ncol=2)
panel_title(ax, "C", "Smaller Bi-NAC beats larger GRPO",
            "Parameter efficiency (Fig. 4, right)")

# (d) Hard competition math - AIME (Qwen3-1.7B)
ax = axes[1, 1]
bm = ["AIME 2024", "AIME 2025", "BeyondAIME"]
t1 = [6.7, 6.7, 2.0]
t2 = [13.3, 10.0, 4.0]
xs = range(3)
ax.bar([x - 0.19 for x in xs], t1, 0.36, color=GRAY)
ax.bar([x + 0.19 for x in xs], t2, 0.36, color=BLUE)
for x, a, b in zip(xs, t1, t2):
    ax.text(x - 0.19, a + 0.3, f"{a:.1f}", ha="center", fontsize=11.5, color=MUTED)
    ax.text(x + 0.19, b + 0.3, f"{b:.1f}", ha="center", fontsize=11.5, color=DARK, fontweight="bold")
ax.set_xticks(list(xs), bm)
ax.set_ylim(0, 16)
ax.set_ylabel("Accuracy (%)")
ax.grid(axis="x", visible=False)
legend_patch(ax, [("Turn 1 (= untrained baseline)", GRAY), ("Bi-NAC after 1 critic turn", BLUE)],
             loc="upper right")
panel_title(ax, "D", "Doubles accuracy on very hard math",
            "Qwen3-1.7B, one refinement turn (Table 1); AIME has 30 problems")

fig.text(0.04, 0.015, "Source: Singh et al., RL with Learnable Textual Feedback: A Bilevel Approach, ICML 2026.",
         fontsize=11, color=FAINT)
fig.savefig("bi_nac_eval_1.png", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------- Figure 2
fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=200)
fig.subplots_adjust(left=0.2, right=0.97, top=0.84, bottom=0.08, hspace=0.62, wspace=0.62)
fig.text(0.04, 0.95, "Evaluation results (2/2): why does it work, and at what cost?",
         fontsize=24, fontweight="bold", color=DARK)

# (e) Ablation - what gets trained
ax = axes[0, 0]
names = ["Hier-NFT  (neither trained)", "Hier-FT  (critic only)", "SCoRe  (actor only)",
         "No bilevel term  (both, uncoupled)", "Bi-NAC  (both, coupled)"]
acc = [35.4, 38.0, 39.6, 41.8, 46.6]
fo = [2.1, 3.0, 2.1, 3.6, 4.2]
cols = [GRAY] * 4 + [BLUE]
ax.barh(range(5), acc, color=cols, height=0.62)
for i, (a, f) in enumerate(zip(acc, fo)):
    ax.text(a + 0.6, i, f"{a:.1f}%   ·   FO {f}", va="center", fontsize=12, color=DARK,
            fontweight="bold" if i == 4 else "normal")
ax.set_yticks(range(5), names)
ax.set_xlim(0, 62)
ax.set_xlabel("MATH-500 accuracy (%)")
ax.grid(axis="y", visible=False)
ax.annotate("", xy=(46.6, 4.42), xytext=(41.8, 4.42),
            arrowprops=dict(arrowstyle="-", color=DARK, lw=0))
panel_title(ax, "E", "The bilevel coupling adds +4.8 points",
            "Ablation, LLaMA-3.2-1B; FO = feedback actionability, 1–5 (Table 4)")

# (f) Multi-turn refinement at inference
ax = axes[0, 1]
turns = [1, 2, 3, 4, 5]
base = [62.4, 65.6, 62.2, 64.7, 63.3]
bn = [62.4, 70.2, 71.6, 71.0, 70.8]
ax.plot(turns, base, color=GRAY, lw=2.5, marker="o", ms=8, markeredgecolor="white", mew=1.5)
ax.plot(turns, bn, color=BLUE, lw=2.5, marker="o", ms=8, markeredgecolor="white", mew=1.5)
ax.text(5.12, bn[-1], "Bi-NAC", va="center", fontsize=12.5, color=DARK, fontweight="bold")
ax.text(5.12, base[-1], "Untrained\nmulti-turn", va="center", fontsize=12, color=MUTED)
ax.annotate("+7.8", xy=(2, 70.2), xytext=(2, 72.6), ha="center", fontsize=12, color=DARK)
ax.annotate("peak 71.6", xy=(3, 71.6), xytext=(3, 73.6), ha="center", fontsize=12, color=DARK)
ax.set_xticks(turns, [f"Turn {t}" for t in turns])
ax.set_xlim(0.7, 5.9)
ax.set_ylim(58, 76)
ax.set_ylabel("MATH-500 accuracy (%)")
panel_title(ax, "F", "Gains compound to turn 3, then hold steady",
            "Qwen3-1.7B, extra refinement turns at inference (Table 3)")

# (g) Compute trade-off - stat tiles (different units, so no shared axis)
ax = axes[1, 0]
ax.axis("off")
panel_title(ax, "G", "Compute: Bi-NAC 2B (1B + 1B) vs. GRPO 3B",
            "Table 2; different units, so shown as tiles")
tiles = [
    ("Tokens per sample", "5,120", "vs. 16,384", "3.2× fewer"),
    ("Training time", "16.5 h", "vs. 18.5 h", "11% less"),
    ("MATH-500 accuracy", "46.6%", "vs. 41.4%", "+5.1 points"),
    ("Inference time", "9.7 s", "vs. 8.3 s", "17% slower"),
]
from matplotlib.patches import FancyBboxPatch
for i, (lab, big, vs, note) in enumerate(tiles):
    cx = (i % 2) * 0.52
    cy = 0.52 - (i // 2) * 0.52
    worse = i == 3
    ax.add_patch(FancyBboxPatch((cx, cy), 0.48, 0.44, boxstyle="round,pad=0,rounding_size=0.03",
                                transform=ax.transAxes, facecolor="#F5F5F3", edgecolor="#E2E1DC", clip_on=False))
    ax.text(cx + 0.03, cy + 0.36, lab, transform=ax.transAxes, fontsize=12.5, color=MUTED)
    ax.text(cx + 0.03, cy + 0.15, big, transform=ax.transAxes, fontsize=26, fontweight="bold",
            color=DARK)
    ax.text(cx + 0.27, cy + 0.17, vs, transform=ax.transAxes, fontsize=12.5, color=MUTED)
    ax.text(cx + 0.03, cy + 0.04, ("▼ " if worse else "▲ ") + note, transform=ax.transAxes,
            fontsize=12.5, fontweight="bold", color="#B3261E" if worse else "#1B7F3B")

# (h) Robustness - model pairings and single-model variant
ax = axes[1, 1]
names = ["Critic Qwen · Actor Qwen", "Critic LLaMA · Actor Qwen",
         "Critic Qwen · Actor LLaMA", "Critic LLaMA · Actor LLaMA",
         "One 1B model, both roles", "Two 1B models (2B total)"]
vals = [48.7, 47.9, 46.8, 44.6, 46.84, 46.56]
ypos = [6.6, 5.6, 4.6, 3.6, 1.6, 0.6]
ax.barh(ypos, vals, color=BLUE, height=0.62)
for y, v in zip(ypos, vals):
    ax.text(v + 0.6, y, f"{v:.1f}%", va="center", fontsize=12, color=DARK)
ax.axvline(39.8, color=MUTED, lw=1.5, ls=(0, (4, 3)))
ax.text(39.3, 7.35, "best baseline (SCoRe, 1B): 39.8", ha="right", fontsize=11, color=MUTED, va="center")
ax.set_yticks(ypos, names)
ax.set_xlim(0, 58)
ax.set_ylim(0, 7.8)
ax.set_xlabel("MATH-500 accuracy (%)")
ax.grid(axis="y", visible=False)
panel_title(ax, "H", "Robust across model families and setups",
            "Mixed LLaMA/Qwen pairs; single-model variant (Table 5, Fig. 5)")

fig.text(0.04, 0.015, "Source: Singh et al., RL with Learnable Textual Feedback: A Bilevel Approach, ICML 2026.",
         fontsize=11, color=FAINT)
fig.savefig("bi_nac_eval_2.png", facecolor="white")
plt.close(fig)
print("ok")
