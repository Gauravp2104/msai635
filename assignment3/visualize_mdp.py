import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from mdp import behavior_policy, target_policy

FIGURES_DIR = "figures"

# Transition table, mirroring the dynamics hard-coded in generate_episode():
# (state, action) -> (next_state, reward)
TRANSITIONS = {
    ("A", "Left"): ("B", 0),
    ("A", "Right"): ("Terminal", 1),
    ("B", "Left"): ("Terminal", 2),
    ("B", "Right"): ("Terminal", 0),
}

# node positions on the canvas
POS = {"A": (0.0, 0.0), "B": (4.0, 2.2), "Terminal": (4.0, -2.2)}

# curvature per edge, so the two B -> Terminal arrows don't overlap
CURVE = {
    ("A", "Left"): 0.0,
    ("A", "Right"): 0.0,
    ("B", "Left"): -0.5,
    ("B", "Right"): 0.5,
}

TARGET_COLOR = "tab:blue"   # actions the target policy takes (pi = 1)
OTHER_COLOR = "gray"        # actions only the behavior policy takes (pi = 0)


def draw_nodes(ax):
    patches = {}
    for s in ("A", "B"):
        x, y = POS[s]
        patches[s] = Circle((x, y), 0.55, facecolor="white", edgecolor="black", lw=2, zorder=3)
        ax.add_patch(patches[s])
        ax.text(x, y, s, ha="center", va="center", fontsize=20, fontweight="bold", zorder=4)

    # terminal drawn as a shaded rounded box to set it apart from regular states
    x, y = POS["Terminal"]
    patches["Terminal"] = FancyBboxPatch((x - 0.75, y - 0.45), 1.5, 0.9, boxstyle="round,pad=0.05",
                                         facecolor="lightgray", edgecolor="black", lw=2, zorder=3)
    ax.add_patch(patches["Terminal"])
    ax.text(x, y, "Terminal", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)

    # start marker pointing into A
    ax.annotate("start", xy=(POS["A"][0] - 0.58, 0), xytext=(-1.7, 0), ha="center", va="center", fontsize=11,
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="black", mutation_scale=16))
    return patches


def draw_edges(ax, patches):
    for (s, a), (s_next, r) in TRANSITIONS.items():
        pi, b = target_policy[s][a], behavior_policy[s][a]
        color = TARGET_COLOR if pi > 0 else OTHER_COLOR
        rad = CURVE[(s, a)]

        arrow = FancyArrowPatch(POS[s], POS[s_next], patchA=patches[s], patchB=patches[s_next],
                                connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=20,
                                lw=2.5 if pi > 0 else 1.8, linestyle="-" if pi > 0 else "--",
                                color=color, zorder=2)
        ax.add_patch(arrow)

        # label at the midpoint of the (possibly curved) arrow: for arc3 the
        # curve midpoint is the chord midpoint shifted by rad/2 * (dy, -dx)
        (x1, y1), (x2, y2) = POS[s], POS[s_next]
        dx, dy = x2 - x1, y2 - y1
        mx = (x1 + x2) / 2 + 0.5 * rad * dy
        my = (y1 + y2) / 2 - 0.5 * rad * dx
        label = f"{a}, r = {r}\nb = {b:g},  π = {pi:g}"
        ax.text(mx, my, label, ha="center", va="center", fontsize=10, color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, lw=1))


def plot_mdp(path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    patches = draw_nodes(ax)
    draw_edges(ax, patches)

    # legend for the two edge styles
    ax.plot([], [], color=TARGET_COLOR, lw=2.5, label="taken by target policy π (always Left)")
    ax.plot([], [], color=OTHER_COLOR, lw=1.8, linestyle="--", label="taken only by behavior policy b")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=10)

    ax.set_xlim(-2.2, 6.6)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("MDP: states, actions, rewards and policy probabilities", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, "mdp_diagram.png")
    plot_mdp(path)
    print(f"Saved {path}")
