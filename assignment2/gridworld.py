import numpy as np

# Action indices -> (row delta, col delta).
# Row 0 is the top of the grid, so "up" decreases the row index.
ACTIONS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


class GridWorld:
    """
    4x4 gridworld (Sutton & Barto, Example 4.1).

    States are numbered row-major 0..15:
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15

    - States 0 (top-left) and 15 (bottom-right) are terminal; the other 14
      are non-terminal.
    - Four actions: up, down, left, right.
    - An action that would move the agent off the grid leaves it in place.
    - Every transition from a non-terminal state yields reward -1.
    - Terminal states end the episode and have value 0.
    """

    def __init__(self, size=4):
        # --- Grid representation ---
        self.size = size
        self.n_states = size * size
        self.n_actions = len(ACTIONS)
        # grid[row, col] holds the state number at that cell
        self.grid = np.arange(self.n_states).reshape(size, size)

        # --- Terminal / non-terminal states ---
        # terminal: top-left corner (s0) and bottom-right corner (s15)
        self.terminal_states = {0, self.n_states - 1}
        self.non_terminal_states = [s for s in range(self.n_states) if s not in self.terminal_states]

    def is_terminal(self, state):
        return state in self.terminal_states

    # --- Conversion between state numbers and (row, col) positions ---
    def to_coords(self, state):
        # state = row * size + col  ->  (row, col)
        return divmod(state, self.size)

    def to_state(self, row, col):
        return row * self.size + col

    # --- Transition dynamics ---
    def step(self, state, action):
        """
        Returns (next_state, reward, done) for a state-action pair.
        Dynamics are deterministic: each (s, a) has exactly one outcome.
        """
        if self.is_terminal(state):
            # terminal states are absorbing: no movement, no reward, episode over
            return state, 0.0, True

        row, col = self.to_coords(state)
        d_row, d_col = ACTIONS[action]
        new_row, new_col = row + d_row, col + d_col

        # an action that would leave the grid keeps the agent where it is
        if not (0 <= new_row < self.size and 0 <= new_col < self.size):
            new_row, new_col = row, col

        next_state = self.to_state(new_row, new_col)
        reward = -1.0  # every transition from a non-terminal state costs -1
        done = self.is_terminal(next_state)  # episode ends on reaching a terminal state
        return next_state, reward, done

    def get_next_state_and_reward(self, state, action):
        # convenience wrapper: just (next_state, reward) for any state-action pair
        next_state, reward, _ = self.step(state, action)
        return next_state, reward

    def transitions(self, state, action):
        """
        Model p(s', r | s, a) as a list of (probability, next_state, reward),
        the form used by the Bellman equation. Deterministic -> one outcome
        with probability 1.
        """
        next_state, reward = self.get_next_state_and_reward(state, action)
        return [(1.0, next_state, reward)]


def uniform_random_policy(env):
    # pi(a|s) = 1/4 for every state and action
    return np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)


if __name__ == "__main__":
    env = GridWorld()

    print("Grid layout (state numbers):")
    print(env.grid)
    print(f"Terminal states:     {sorted(env.terminal_states)}")
    print(f"Non-terminal states: {env.non_terminal_states}\n")

    # transition table: for every state-action pair, the next state and reward
    header = f"{'State':>5} | " + " | ".join(f"{ACTION_NAMES[a]:>8}" for a in range(env.n_actions))
    print(header)
    print("-" * len(header))
    for s in range(env.n_states):
        cells = []
        for a in range(env.n_actions):
            next_state, reward = env.get_next_state_and_reward(s, a)
            cells.append(f"{next_state:>3},{reward:>4.0f}")
        mark = "  (terminal)" if env.is_terminal(s) else ""
        print(f"{s:>5} | " + " | ".join(f"{c:>8}" for c in cells) + mark)
    print("-" * len(header))
    print("Each cell: next_state, reward")
