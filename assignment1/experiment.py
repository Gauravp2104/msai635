from datetime import datetime

import numpy as np

from agents import EpsilonGreedyAgent, GreedyAgent
from bandit import KArmedBandit


def run_experiment(agent_factory, k=10, steps=1000, runs=200):
    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))

    for run in range(runs):
        bandit = KArmedBandit(k=k)
        agent = agent_factory(k)
        optimal = bandit.optimal_action()

        for t in range(steps):
            action = agent.select_action()
            reward = bandit.step(action)
            agent.update(action, reward)

            rewards[run, t] = reward
            optimal_actions[run, t] = 1 if action == optimal else 0

    avg_reward = rewards.mean(axis=0)
    pct_optimal = optimal_actions.mean(axis=0) * 100
    return avg_reward, pct_optimal


if __name__ == "__main__":
    STEPS = 1000
    RUNS = 200
    EPSILONS = [0.0, 0.01, 0.10]

    run_timestamp = datetime.now()

    # agent_factory: k -> agent instance
    experiments = [("Greedy (epsilon=0.0)", lambda k: GreedyAgent(k=k))]
    for eps in EPSILONS:
        if eps == 0.0:
            continue  # equivalent to the pure greedy agent above
        experiments.append(
            (f"Epsilon-greedy (epsilon={eps})", lambda k, eps=eps: EpsilonGreedyAgent(k=k, epsilon=eps))
        )

    results = {}
    for label, agent_factory in experiments:
        results[label] = run_experiment(agent_factory, k=10, steps=STEPS, runs=RUNS)

    checkpoints = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    print(f"Run started: {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    for label, (avg_reward, pct_optimal) in results.items():
        header = f"{'Step':>6} | {'Avg Reward':>12} | {'% Optimal Action':>17}"
        print(f"\n{label} — {RUNS} runs x {STEPS} steps\n")
        print(header)
        print("-" * len(header))
        for step in checkpoints:
            print(f"{step:>6} | {avg_reward[step - 1]:>12.4f} | {pct_optimal[step - 1]:>16.2f}%")

    import matplotlib.pyplot as plt

    colors = ["tab:blue", "tab:green", "tab:orange", "tab:red"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for (label, (avg_reward, pct_optimal)), color in zip(results.items(), colors):
        ax1.plot(avg_reward, color=color, label=label)
        ax2.plot(pct_optimal, color=color, label=label)

    ax1.set_xlabel("Time steps")
    ax1.set_ylabel("Average reward")
    ax1.set_title(f"Agent reward vs. time step ({RUNS} runs, {STEPS} steps)")
    ax1.legend()

    ax2.set_xlabel("Time steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_title(f"% Optimal action vs. time step ({RUNS} runs, {STEPS} steps)")
    ax2.set_ylim(0, 100)
    ax2.legend()

    fig.suptitle(f"Greedy vs. epsilon-greedy agents — generated {run_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    plt.tight_layout()

    out_path = f"epsilon_greedy_agent_performance_{run_timestamp.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
