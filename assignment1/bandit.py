import numpy as np

class KArmedBandit:
    def __init__(self, k=10):
        self.k = k
        # true action values q*(a) sampled from N(0, 1)
        self.q_true = np.random.normal(loc=0.0, scale=1.0, size=self.k)
        # arm with the highest true action value
        self._optimal_action = np.argmax(self.q_true)

    def step(self, action):
        # reward sampled from N(q*(a), 1)
        return np.random.normal(loc=self.q_true[action], scale=1.0)

    def optimal_action(self):
        return self._optimal_action


if __name__ == "__main__":
    bandit = KArmedBandit(k=10)
    optimal = bandit.optimal_action()

    header = f"{'Arm':>4} | {'q_true(a)':>10} | {'Reward':>10} | {'Optimal':>7}"
    print(header)
    print("-" * len(header))

    for action in range(bandit.k):
        reward = bandit.step(action)
        mark = "*" if action == optimal else ""
        print(f"{action:>4} | {bandit.q_true[action]:>10.4f} | {reward:>10.4f} | {mark:>7}")

    print("-" * len(header))
    print(f"Optimal action: {optimal} (q_true = {bandit.q_true[optimal]:.4f})")
