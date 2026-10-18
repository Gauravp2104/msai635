import os

import matplotlib.pyplot as plt
import numpy as np

from gridworld import ACTIONS, GridWorld, uniform_random_policy
from policy_evaluation import iterative_policy_evaluation, solve_bellman_linear
from value_iteration import extract_greedy_policy, print_policy_arrows, print_policy_labels, value_iteration

GAMMA = 1.0
THETA = 1e-4
FIGURES_DIR = "figures"


def print_grid(V, size, title):
    print(title)
    for row in V.reshape(size, size):
        print("  " + " ".join(f"{v:>7.2f}" for v in row))
    print()


def plot_value_table(env, V, title, path, cmap="viridis"):
    """
    Save V as a formatted 4x4 table: one cell per state, colored by value,
    labeled with the state number and its value. Terminal cells are marked T.
    """
    grid = V.reshape(env.size, env.size)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(grid, cmap=cmap)

    # midpoint of the color range decides black vs white text for readability
    mid = (grid.min() + grid.max()) / 2
    for r in range(env.size):
        for c in range(env.size):
            s = env.to_state(r, c)
            label = f"s{s}" + (" (T)" if env.is_terminal(s) else "")
            color = "white" if grid[r, c] < mid else "black"
            ax.text(c, r - 0.18, label, ha="center", va="center", fontsize=9, color=color)
            ax.text(c, r + 0.12, f"{grid[r, c]:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)

    draw_cell_borders(ax, env.size)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="state value")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_policy(env, best_actions, V_star, path):
    """
    Save the optimal policy as a 4x4 grid with an arrow for every optimal
    action in each state (several arrows = tied actions). Terminal cells are
    shaded and marked T; each cell also shows its v*(s) in the corner.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)  # row 0 at the top, like the grid
    ax.set_aspect("equal")

    for s in range(env.n_states):
        r, c = env.to_coords(s)
        if env.is_terminal(s):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="lightgray"))
            ax.text(c, r, "T", ha="center", va="center", fontsize=20, fontweight="bold")
            continue

        # one arrow per optimal action, pointing from the cell center
        for a in best_actions[s]:
            d_row, d_col = ACTIONS[a]
            ax.annotate("", xy=(c + 0.35 * d_col, r + 0.35 * d_row), xytext=(c, r),
                        arrowprops=dict(arrowstyle="-|>", lw=2, color="tab:blue", mutation_scale=18))
        ax.text(c - 0.45, r - 0.45, f"s{s}", ha="left", va="top", fontsize=8, color="dimgray")
        ax.text(c + 0.45, r + 0.45, f"{V_star[s]:.0f}", ha="right", va="bottom", fontsize=8, color="dimgray")

    draw_cell_borders(ax, env.size)
    ax.set_title("Optimal policy pi* (from value iteration)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gridworld(env, path):
    """
    Save the environment layout: state numbers, shaded terminal states, the
    four actions, and an example of the "bump into the wall -> stay" rule.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.set_aspect("equal")

    for s in range(env.n_states):
        r, c = env.to_coords(s)
        if env.is_terminal(s):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="lightgray"))
            ax.text(c, r - 0.1, f"s{s}", ha="center", va="center", fontsize=14, fontweight="bold")
            ax.text(c, r + 0.22, "terminal\nv = 0", ha="center", va="center", fontsize=8)
        else:
            ax.text(c, r, f"s{s}", ha="center", va="center", fontsize=14)

    # the four actions drawn from s6, each costing -1
    r, c = env.to_coords(6)
    for a, (d_row, d_col) in ACTIONS.items():
        ax.annotate("", xy=(c + 0.42 * d_col, r + 0.42 * d_row), xytext=(c + 0.18 * d_col, r + 0.18 * d_row),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color="tab:blue"))
    ax.text(c + 0.3, r - 0.3, "-1", color="tab:blue", fontsize=8, ha="center", va="center")

    # wall rule: "up" from s2 bumps into the edge and stays in s2
    r, c = env.to_coords(2)
    ax.annotate("", xy=(c + 0.2, r - 0.12), xytext=(c - 0.2, r - 0.12),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="tab:red", connectionstyle="arc3,rad=-1.6"))
    ax.text(c, r + 0.3, "up: stays", color="tab:red", fontsize=8, ha="center", va="center")

    draw_cell_borders(ax, env.size)
    ax.set_title("4x4 gridworld (reward -1 per step, gamma = 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(curves, theta, path):
    """
    Save max |V_{k+1} - V_k| per sweep on a log scale, one line per method,
    with the stopping threshold theta drawn as a dashed line.
    """
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for label, history in curves.items():
        H = np.array(history)
        deltas = np.abs(np.diff(H, axis=0)).max(axis=1)
        k = np.arange(1, len(deltas) + 1)
        nonzero = deltas > 0
        # value iteration reaches delta = 0 exactly, which a log axis can't show: mark it instead
        line, = ax.semilogy(k[nonzero], deltas[nonzero], marker="o" if len(deltas) < 10 else None,
                            ms=4, label=f"{label} ({len(deltas)} sweeps)")
        if not nonzero.all():
            ax.annotate(f"delta = 0 at sweep {k[~nonzero][0]}", xy=(k[nonzero][-1], deltas[nonzero][-1]),
                        xytext=(12, 2e-3), fontsize=8, color=line.get_color(),
                        arrowprops=dict(arrowstyle="->", color=line.get_color()))
    ax.axhline(theta, color="black", ls="--", lw=1, label=f"theta = {theta:g}")
    ax.set_ylim(5e-5, 5)
    ax.set_xlabel("sweep k")
    ax.set_ylabel(r"$\max_s\,|V_k(s) - V_{k-1}(s)|$")
    ax.set_title("Convergence of the iterative updates")
    ax.legend(fontsize=8)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_cell_borders(ax, size):
    # thin black lines between cells so the image reads as a table
    for k in range(size + 1):
        ax.axhline(k - 0.5, color="black", lw=1)
        ax.axvline(k - 0.5, color="black", lw=1)
    ax.set_xticks([])
    ax.set_yticks([])


if __name__ == "__main__":
    env = GridWorld(size=4)
    policy = uniform_random_policy(env)

    # iterative policy evaluation, two-array version (matches the textbook's k-sweep figure)
    V_sync, iters_sync, history = iterative_policy_evaluation(env, policy, gamma=GAMMA, theta=THETA, in_place=False)
    for k in [0, 1, 2, 3, 10]:
        print_grid(history[k], env.size, f"V_k after k = {k} sweeps:")
    print(f"theta = {THETA:g}")
    print(f"Iterations until convergence (two-array): {iters_sync}")
    print_grid(V_sync, env.size, "Final value table (two-array):")

    # in-place version typically converges in fewer sweeps
    V_inplace, iters_inplace, history_inplace = iterative_policy_evaluation(env, policy, gamma=GAMMA, theta=THETA, in_place=True)
    print(f"Iterations until convergence (in-place): {iters_inplace}")
    print_grid(V_inplace, env.size, "Final value table (in-place):")

    # exact solution from the linear Bellman system, used as a check
    V_exact = solve_bellman_linear(env, policy, gamma=GAMMA)
    print_grid(V_exact, env.size, "Exact solution of (I - gamma * P_pi) v = r_pi:")

    print(f"max |V_iterative - V_exact| (two-array): {np.max(np.abs(V_sync - V_exact)):.2e}")
    print(f"max |V_iterative - V_exact| (in-place):  {np.max(np.abs(V_inplace - V_exact)):.2e}")


    # value iteration: Bellman optimality update -> v* and the optimal policy
    V_star, iters_vi, history_vi = value_iteration(env, gamma=GAMMA, theta=THETA)
    print(f"Iterations until convergence (value iteration): {iters_vi}")
    print_grid(V_star, env.size, "Optimal value function v*(s):")

    _, best_actions = extract_greedy_policy(env, V_star, gamma=GAMMA)
    print("Optimal policy pi*(s) (several arrows = tied optimal actions, T = terminal):")
    print_policy_arrows(env, best_actions)
    print("\nOptimal policy as text labels:")
    print_policy_labels(env, best_actions)

    # --- visualizations ---
    os.makedirs(FIGURES_DIR, exist_ok=True)
    outputs = {
        "gridworld_layout.png": lambda p: plot_gridworld(env, p),
        "convergence.png": lambda p: plot_convergence({
            "policy evaluation, two-array": history,
            "policy evaluation, in-place": history_inplace,
            "value iteration": history_vi,
        }, THETA, p),
        "policy_evaluation_values.png": lambda p: plot_value_table(
            env, V_sync, f"v_pi: uniform random policy\n(policy evaluation, {iters_sync} iterations)", p),
        "optimal_values.png": lambda p: plot_value_table(
            env, V_star, f"v*: optimal state values\n(value iteration, {iters_vi} iterations)", p, cmap="magma"),
        "optimal_policy.png": lambda p: plot_policy(env, best_actions, V_star, p),
    }
    print()
    for name, make_figure in outputs.items():
        path = os.path.join(FIGURES_DIR, name)
        make_figure(path)
        print(f"Saved {path}")
