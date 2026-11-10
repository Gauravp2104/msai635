"""
Tabular Dyna-Q agent

Each real time step does three things:
  (d) direct RL  - one Q-learning update from the real transition
  (e) model learning - remember Model(S, A) = (R, S') (the maze is deterministic)
  (f) planning   - n times: pick a previously observed (S, A) at random, ask the
                   model what happens, and apply the same Q-learning update to
                   that simulated transition
"""

import numpy as np

from maze import ChangingMaze

GAMMA = 0.95
ALPHA = 1.0
EPSILON = 0.1
N_PLANNING = 10
N_STEPS = 3000
KAPPA = 0.001
SHORTCUT = (3, 8)


class DynaQAgent:
    name = "Dyna-Q"

    def __init__(self, n_actions, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON,
                 n_planning=N_PLANNING, seed=None):
        self.n_actions = n_actions
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.n_planning = n_planning
        self.rng = np.random.default_rng(seed)

        self.Q = {}                 # state -> np.array of action values (lazily zero-initialised)
        self.model = {}             # (state, action) -> (reward, next_state, terminal)
        self.model_keys = []        # observed (state, action) pairs, for O(1) uniform sampling

    # ------------------------------------------------------------ Q helpers
    def q(self, state):
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def greedy(self, state):
        """Argmax with random tie-breaking (important early on, when all Q are 0)."""
        values = self.q(state)
        best = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best))

    def act(self, state):
        """epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(state)

    def q_update(self, s, a, r, s2, terminal):
        """One-step Q-learning: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]."""
        target = r if terminal else r + self.gamma * self.q(s2).max()
        self.q(s)[a] += self.alpha * (target - self.q(s)[a])

    # ------------------------------------------------------ model / planning
    def update_model(self, s, a, r, s2, terminal, t):
        if (s, a) not in self.model:
            self.model_keys.append((s, a))
        self.model[(s, a)] = (r, s2, terminal)

    def planning_reward(self, s, a, r, t):
        """Reward used for a simulated transition. Dyna-Q+ overrides this to add its bonus."""
        return r

    def plan(self, t):
        for _ in range(self.n_planning):
            s, a = self.model_keys[self.rng.integers(len(self.model_keys))]
            r, s2, terminal = self.model[(s, a)]
            self.q_update(s, a, self.planning_reward(s, a, r, t), s2, terminal)

    # --------------------------------------------------------- one real step
    def learn(self, s, a, r, s2, terminal, t):
        """Called once per real time step t, after the environment has responded."""
        self.q_update(s, a, r, s2, terminal)       # (d) direct RL
        self.update_model(s, a, r, s2, terminal, t)  # (e) model learning
        self.plan(t)                                # (f) planning


class DynaQPlusAgent(DynaQAgent):
    """
    Dyna-Q+ (Sutton & Barto 2018, Section 8.3).

    Identical to Dyna-Q except:
      * tau(s, a) = time steps since (s, a) was last executed in the real environment;
        planning updates use the reward r + kappa * sqrt(tau). Real (direct-RL)
        updates still use the true reward, so the bonus only acts through planning.
      * The first time a state s is visited, all actions from s are put in the model
        as "returns to s with reward 0", last tried at t = 0. Their bonus therefore
        grows from the start of the run until the action is actually taken, at which
        point the real transition overwrites the placeholder.
    """
    name = "Dyna-Q+"

    def __init__(self, n_actions, kappa=KAPPA, **kwargs):
        super().__init__(n_actions, **kwargs)
        self.kappa = kappa
        self.last_tried = {}        # (state, action) -> last real time step it was executed
        self.seen_states = set()

    def init_state(self, s):
        """Footnote to Example 8.3: untried actions are modelled as self-loops with reward 0."""
        if s in self.seen_states:
            return
        self.seen_states.add(s)
        for a in range(self.n_actions):
            if (s, a) not in self.model:
                self.model_keys.append((s, a))
                self.model[(s, a)] = (0.0, s, False)
                self.last_tried[(s, a)] = 0

    def update_model(self, s, a, r, s2, terminal, t):
        self.init_state(s)
        super().update_model(s, a, r, s2, terminal, t)
        self.last_tried[(s, a)] = t

    def planning_reward(self, s, a, r, t):
        tau = t - self.last_tried[(s, a)]
        return r + self.kappa * np.sqrt(tau)


def run(agent, env=None, n_steps=N_STEPS, return_trips=False):
    """
    Run `agent` for `n_steps` real time steps on a fresh ChangingMaze.
    Returns the cumulative reward after every step (length n_steps). With
    return_trips=True also returns one (t_end, length, via_shortcut) tuple per
    trip from S to G, where via_shortcut means the trip passed through (3, 8).
    """
    env = env or ChangingMaze()
    state = env.reset()
    cumulative = np.zeros(n_steps)
    total = 0.0
    trips, trip_len, via = [], 0, False
    for t in range(n_steps):
        action = agent.act(state)
        next_state, reward, terminal = env.step(action)
        agent.learn(state, action, reward, next_state, terminal, t)
        total += reward
        cumulative[t] = total
        trip_len += 1
        via |= next_state == SHORTCUT
        if terminal:
            trips.append((t, trip_len, via))
            trip_len, via = 0, False
        state = env.state          # S again if the goal was just reached
    return (cumulative, trips) if return_trips else cumulative


def greedy_path(agent, env, max_len=40):
    """
    Follow the agent's greedy policy (ties broken by lowest index) from S on env's current layout.
    For Dyna-Q+ this snapshot can loop: the planning bonus deliberately inflates Q for
    actions not tried recently, so judge its behaviour from real trips instead.
    """
    s, path = env.start, [env.start]
    while s != env.goal and len(path) <= max_len:
        s = env.next_cell(s, int(np.argmax(agent.q(s))))
        path.append(s)
    return path


if __name__ == "__main__":
    env = ChangingMaze()

    # --- Dyna-Q: Q-update and model on hand-made transitions -----------------
    a = DynaQAgent(env.n_actions, n_planning=0, seed=0)
    a.learn((0, 7), 3, 1.0, (0, 8), True, t=0)
    assert a.q((0, 7))[3] == 1.0                         # alpha=1 -> Q = r on terminal step
    a.learn((1, 7), 0, 0.0, (0, 7), False, t=1)
    assert np.isclose(a.q((1, 7))[0], GAMMA)             # 0 + 0.95 * 1
    assert a.model[((1, 7), 0)] == (0.0, (0, 7), False)
    assert len(a.model) == 2                             # Dyna-Q only stores real transitions

    # --- Dyna-Q+: placeholder model entries, tau tracking, bonus ------------
    p = DynaQPlusAgent(env.n_actions, n_planning=0, seed=0)
    p.learn((5, 3), 0, 0.0, (4, 3), False, t=10)
    assert len(p.model) == 4                             # all 4 actions of the new state added
    assert p.model[((5, 3), 0)] == (0.0, (4, 3), False)  # executed action: real transition
    assert p.model[((5, 3), 1)] == (0.0, (5, 3), False)  # untried action: self-loop, r = 0
    assert p.last_tried[((5, 3), 0)] == 10 and p.last_tried[((5, 3), 1)] == 0
    assert np.isclose(p.planning_reward((5, 3), 1, 0.0, t=100), KAPPA * 10)    # tau = 100
    assert p.planning_reward((5, 3), 0, 0.0, t=10) == 0.0                      # tau = 0
    assert p.q((5, 3))[0] == 0.0                         # no bonus in the direct-RL update
    p.learn((5, 3), 1, 0.0, (5, 3), False, t=11)         # executing it refreshes tau
    assert p.last_tried[((5, 3), 1)] == 11 and len(p.model) == 4
    print("unit checks passed\n")

    # --- full runs, same seeds for both agents -------------------------------
    # "last 500": the trips completed in steps 2500-2999, i.e. the agent's settled behaviour
    SEEDS = range(10)
    print(f"alpha={ALPHA}, gamma={GAMMA}, eps={EPSILON}, n={N_PLANNING}, kappa={KAPPA}, {N_STEPS} steps\n")
    print(f"{'agent':>8} {'seed':>4} {'R@500':>6} {'R@3000':>7} {'goals 500-3000':>15} "
          f"{'last-500 trip len':>18} {'last-500 via shortcut':>22}")
    for cls in (DynaQAgent, DynaQPlusAgent):
        for seed in SEEDS:
            cum, trips = run(cls(env.n_actions, seed=seed), ChangingMaze(), return_trips=True)
            late = [(n, v) for t, n, v in trips if t >= N_STEPS - 500]
            print(f"{cls.name:>8} {seed:>4} {cum[499]:>6.0f} {cum[-1]:>7.0f} {cum[-1] - cum[499]:>15.0f} "
                  f"{np.mean([n for n, _ in late]):>18.1f} {sum(v for _, v in late):>14}/{len(late):<7}")
        print()

    # --- larger sample for a summary -----------------------------------------
    n_runs = 100
    print(f"summary over {n_runs} seeds (shortest routes: old 16 steps, shortcut 10 steps)")
    for cls in (DynaQAgent, DynaQPlusAgent):
        cums, via = [], []
        for seed in range(n_runs):
            cum, trips = run(cls(env.n_actions, seed=seed), ChangingMaze(), return_trips=True)
            cums.append(cum)
            late = [v for t, _, v in trips if t >= N_STEPS - 500]
            via.append(np.mean(late))
        cums = np.array(cums)
        print(f"  {cls.name:8s} mean R@500 = {cums[:, 499].mean():4.1f}   mean R@3000 = {cums[:, -1].mean():6.1f}"
              f"   last-500 trips via shortcut = {np.mean(via):.0%}"
              f"   runs mostly using shortcut = {np.mean(np.array(via) > 0.5):.0%}")
