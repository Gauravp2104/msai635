import numpy as np


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


class EpsilonGreedyAgent:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.Q = np.zeros(k)  # action-value estimates
        self.N = np.zeros(k)  # action selection counts

    def select_action(self):
        # with probability epsilon, explore a random action; otherwise exploit
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        max_value = np.max(self.Q)
        best_actions = np.flatnonzero(self.Q == max_value)
        return np.random.choice(best_actions)

    def update(self, action, reward):
        # sample-average update: Q(a) <- Q(a) + (1/N(a)) * (reward - Q(a))
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
