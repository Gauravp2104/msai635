"""
Dyna-Q vs Dyna-Q+ on the changing (shortcut) maze.

Both agents are run for the same seeds on identical, freshly created mazes.
The shortcut opens at `--change-step` (default 500, as the assignment
specifies). The script prints a summary table, saves the raw cumulative-reward
curves, and plots mean cumulative reward vs time step.

Usage (from this folder, with the Anaconda Python):
    python experiments.py                                  # assignment setup: 100 runs, 3000 steps, change at 500
    python experiments.py --runs 30                        # quicker
    python experiments.py --change-step 3000 --steps 6000  # Sutton & Barto timing (supplementary)
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from agents import (ALPHA, EPSILON, GAMMA, KAPPA, N_PLANNING, N_STEPS, SHORTCUT,
                    DynaQAgent, DynaQPlusAgent, run)
from maze import CHANGE_STEP, N_ACTIONS, ChangingMaze

AGENTS = (DynaQAgent, DynaQPlusAgent)
COLORS = {"Dyna-Q": "#2a78d6", "Dyna-Q+": "#eb6834"}   # validated blue / orange pair
FIGURES_DIR = "figures"
RESULTS_DIR = "results"


def run_experiment(n_runs, n_steps, change_step):
    """
    Run every agent class for seeds 0..n_runs-1 (same seeds for each agent).
    Returns {name: {"cum": (n_runs, n_steps) array, "trips": list of per-run trip lists}}.
    """
    results = {}
    for cls in AGENTS:
        cums, trips = [], []
        t0 = time.time()
        for seed in range(n_runs):
            agent = cls(N_ACTIONS, seed=seed)
            cum, tr = run(agent, ChangingMaze(change_step=change_step), n_steps=n_steps, return_trips=True)
            cums.append(cum)
            trips.append(tr)
        results[cls.name] = {"cum": np.array(cums), "trips": trips}
        print(f"  ran {n_runs} x {cls.name:8s} in {time.time() - t0:5.1f}s")
    return results


def summarize(results, n_steps, change_step, window=500):
    """Print reward before/after the change and how often the shortcut is used at the end."""
    print(f"\n{'agent':8s} {'R@change':>9} {'R@end':>8} {'reward after change':>20} "
          f"{'last-%d trips via shortcut' % window:>27} {'runs mostly on shortcut':>24}")
    for name, res in results.items():
        cum = res["cum"]
        before = cum[:, change_step - 1]
        end = cum[:, -1]
        via = [np.mean([v for t, _, v in tr if t >= n_steps - window] or [0.0]) for tr in res["trips"]]
        print(f"{name:8s} {before.mean():9.1f} {end.mean():8.1f} "
              f"{(end - before).mean():12.1f} ± {(end - before).std(ddof=1) / np.sqrt(len(end)):4.1f}  "
              f"{np.mean(via):>26.0%} {np.mean(np.array(via) > 0.5):>24.0%}")

    # Split runs by whether they had already reached G (i.e. knew the old route) before the change.
    # Only the "knew old route" group tests adaptation; the others never had anything to unlearn.
    print(f"\nruns split by whether G was reached before step {change_step}:")
    for name, res in results.items():
        before = res["cum"][:, change_step - 1]
        via = np.array([np.mean([v for t, _, v in tr if t >= n_steps - window] or [0.0]) for tr in res["trips"]])
        for label, mask in (("knew old route", before > 0), ("no goal yet", before == 0)):
            if mask.any():
                print(f"  {name:8s} {label:15s} {mask.sum():3d} runs, switched to shortcut: "
                      f"{np.sum(via[mask] > 0.5):3d}/{mask.sum():<3d} ({np.mean(via[mask] > 0.5):.0%})")


def shortcut_usage(trips_per_run, n_steps, bin_size=100):
    """Per time bin: fraction of completed trips that went through the shortcut, pooled over runs."""
    n_bins = n_steps // bin_size
    total = np.zeros(n_bins)
    via = np.zeros(n_bins)
    for trips in trips_per_run:
        for t, _, v in trips:
            b = min(t // bin_size, n_bins - 1)
            total[b] += 1
            via[b] += v
    with np.errstate(invalid="ignore"):
        frac = via / total
    centers = (np.arange(n_bins) + 0.5) * bin_size
    return centers, frac


def plot_shortcut_usage(results, n_steps, change_step, path, bin_size=100):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, res in results.items():
        x, frac = shortcut_usage(res["trips"], n_steps, bin_size)
        ax.plot(x, 100 * frac, color=COLORS[name], lw=2, marker="o", ms=4, label=name)
    ax.axvline(change_step, color="#6b6b6b", ls="--", lw=1.2)
    ax.annotate(f"shortcut opens (step {change_step})", xy=(change_step, 0.93), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=9, color="#4a4a4a")
    ax.set_xlim(0, n_steps)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Trips via shortcut (%)")
    ax.set_title(f"Share of goal trips passing through (3,8), per {bin_size}-step window")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot(results, n_steps, change_step, path):
    n_runs = next(iter(results.values()))["cum"].shape[0]
    steps = np.arange(1, n_steps + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, res in results.items():
        cum = res["cum"]
        mean = cum.mean(axis=0)
        se = cum.std(axis=0, ddof=1) / np.sqrt(n_runs)
        ax.fill_between(steps, mean - se, mean + se, color=COLORS[name], alpha=0.2, lw=0)
        ax.plot(steps, mean, color=COLORS[name], lw=2, label=name)
        ax.annotate(f"{name}  {mean[-1]:.0f}", xy=(n_steps, mean[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=10, color="#2b2b2b")

    ax.axvline(change_step, color="#6b6b6b", ls="--", lw=1.2)
    ax.annotate(f"shortcut opens\n(step {change_step})", xy=(change_step, 0.95), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", va="top", fontsize=9, color="#4a4a4a")

    ax.set_xlim(0, n_steps)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Cumulative reward")
    ax.set_title(f"Dyna-Q vs Dyna-Q+ in the shortcut maze (mean of {n_runs} runs, ±1 SE)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(color="#e6e6e6", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.text(0.99, 0.01, f"α={ALPHA}, γ={GAMMA}, ε={EPSILON}, n={N_PLANNING}, κ={KAPPA}",
             ha="right", va="bottom", fontsize=8, color="#6b6b6b")
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)   # room for the end-of-line labels

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=int, default=100, help="independent runs (seeds) per agent")
    parser.add_argument("--steps", type=int, default=N_STEPS, help="real time steps per run")
    parser.add_argument("--change-step", type=int, default=CHANGE_STEP, help="time step at which the shortcut opens")
    args = parser.parse_args()

    tag = f"change{args.change_step}_steps{args.steps}_runs{args.runs}"
    print(f"Shortcut maze: {args.runs} runs per agent, {args.steps} steps, shortcut opens at step "
          f"{args.change_step}\nα={ALPHA}, γ={GAMMA}, ε={EPSILON}, n={N_PLANNING}, κ={KAPPA}, "
          f"shortcut cell {SHORTCUT}")

    results = run_experiment(args.runs, args.steps, args.change_step)
    summarize(results, args.steps, args.change_step)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    npz = os.path.join(RESULTS_DIR, f"cumulative_reward_{tag}.npz")
    np.savez(npz, **{name.replace("+", "_plus"): res["cum"] for name, res in results.items()})
    fig = plot(results, args.steps, args.change_step, os.path.join(FIGURES_DIR, f"cumulative_reward_{tag}.png"))
    fig2 = plot_shortcut_usage(results, args.steps, args.change_step,
                               os.path.join(FIGURES_DIR, f"shortcut_usage_{tag}.png"))
    print(f"\nsaved curves to {npz}\nsaved plots  to {fig}\n             and {fig2}")


if __name__ == "__main__":
    main()
