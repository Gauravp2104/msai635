import numpy as np

from bandit import KArmedBandit


class GreedyAgent:
    def __init__(self, k=10):
        self.k = k
        self.Q = np.zeros(k)  # action-value estimates
        self.N = np.zeros(k)  # action selection counts

    def select_action(self):
        # pure exploitation: pick the highest-valued action, breaking ties randomly
        max_value = np.max(self.Q)
        best_actions = np.flatnonzero(self.Q == max_value)
        return np.random.choice(best_actions)

    def update(self, action, reward):
        # sample-average update: Q(a) <- Q(a) + (1/N(a)) * (reward - Q(a))
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]


def run_experiment(k=10, steps=1000, runs=200):
    rewards = np.zeros((runs, steps))
    optimal_actions = np.zeros((runs, steps))

    for run in range(runs):
        bandit = KArmedBandit(k=k)
        agent = GreedyAgent(k=k)
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

    avg_reward, pct_optimal = run_experiment(k=10, steps=STEPS, runs=RUNS)

    checkpoints = [1, 10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    header = f"{'Step':>6} | {'Avg Reward':>12} | {'% Optimal Action':>17}"
    print(f"Greedy agent — {RUNS} runs x {STEPS} steps\n")
    print(header)
    print("-" * len(header))
    for step in checkpoints:
        print(f"{step:>6} | {avg_reward[step - 1]:>12.4f} | {pct_optimal[step - 1]:>16.2f}%")

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(avg_reward, color="tab:blue")
    ax1.set_ylabel("Average reward")
    ax1.set_title(f"Greedy agent performance ({RUNS} runs, {STEPS} steps)")

    ax2.plot(pct_optimal, color="tab:green")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("% Optimal action")
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    out_path = "greedy_agent_performance.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")
