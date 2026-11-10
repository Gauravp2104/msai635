"""
Grid: 6 rows x 9 columns, 0-indexed, row 0 at the top.
  Start S = (5, 3)
  Goal  G = (0, 8)
  Reward: +1 on the transition into G, 0 otherwise.
  Walls before step 500: row 3, columns 1..8  (only column 0 open)
  Walls after  step 500: row 3, columns 1..7  (column 0 and the new shortcut at column 8 open)
"""

ROWS, COLS = 6, 9
START = (5, 3)
GOAL = (0, 8)
CHANGE_STEP = 500

# actions as (d_row, d_col); row 0 is the top, so "up" decreases the row index
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["up", "down", "left", "right"]
N_ACTIONS = len(ACTIONS)

WALLS_BEFORE = frozenset((3, c) for c in range(1, 9))  # (3,1) .. (3,8)
WALLS_AFTER = frozenset((3, c) for c in range(1, 8))   # (3,1) .. (3,7)


class ChangingMaze:
    """6x9 gridworld whose wall opens a shortcut after `change_step` time steps."""

    def __init__(self, change_step=CHANGE_STEP):
        self.rows, self.cols = ROWS, COLS
        self.start, self.goal = START, GOAL
        self.change_step = change_step
        self.n_actions = N_ACTIONS
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Restore the original maze, put the agent at S and zero the clock."""
        self.t = 0
        self.walls = WALLS_BEFORE
        self.state = self.start
        return self.state

    @property
    def changed(self):
        return self.walls is WALLS_AFTER

    def open_shortcut(self):
        self.walls = WALLS_AFTER

    def states(self):
        """All non-wall cells under the current layout."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.walls]

    # -------------------------------------------------------------- dynamics
    def next_cell(self, state, action):
        """Deterministic move: bumping into a wall or the border leaves the agent in place."""
        dr, dc = ACTIONS[action]
        r, c = state[0] + dr, state[1] + dc
        if not (0 <= r < self.rows and 0 <= c < self.cols) or (r, c) in self.walls:
            return state
        return (r, c)

    def step(self, action):
        """
        Take one real time step.

        Returns (next_state, reward, reached_goal). On reaching G the returned
        next_state is G (so the agent can treat it as terminal in its update),
        while the environment itself has already moved the agent back to S.
        """
        next_state = self.next_cell(self.state, action)
        reached_goal = next_state == self.goal
        reward = 1.0 if reached_goal else 0.0

        self.state = self.start if reached_goal else next_state

        self.t += 1
        if self.t == self.change_step:
            self.open_shortcut()

        return next_state, reward, reached_goal

    # ------------------------------------------------------------- rendering
    def render(self, agent=True):
        """ASCII picture of the current layout: '#' wall, S start, G goal, A agent."""
        lines = ["    " + " ".join(str(c) for c in range(self.cols))]
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                cell = (r, c)
                if agent and cell == self.state and cell != self.start:
                    row.append("A")
                elif cell in self.walls:
                    row.append("#")
                elif cell == self.start:
                    row.append("S")
                elif cell == self.goal:
                    row.append("G")
                else:
                    row.append(".")
            lines.append(f"{r:>2}  " + " ".join(row))
        return "\n".join(lines)


def plot_layouts(path="figures/maze_layouts.png"):
    """Side-by-side figure of the maze before and after the shortcut opens."""
    import os

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "#bfbfbf", "#a8d08d", "#ffd966"])  # open, wall, S, G
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, walls, title in [
        (axes[0], WALLS_BEFORE, f"Before step {CHANGE_STEP}"),
        (axes[1], WALLS_AFTER, f"From step {CHANGE_STEP} (shortcut at (3,8))"),
    ]:
        grid = [[0] * COLS for _ in range(ROWS)]
        for r, c in walls:
            grid[r][c] = 1
        grid[START[0]][START[1]] = 2
        grid[GOAL[0]][GOAL[1]] = 3
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)
        ax.text(START[1], START[0], "S", ha="center", va="center", fontweight="bold")
        ax.text(GOAL[1], GOAL[0], "G", ha="center", va="center", fontweight="bold")
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.set_xticks([x - 0.5 for x in range(1, COLS)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, ROWS)], minor=True)
        ax.grid(which="minor", color="black", lw=0.8)
        ax.tick_params(which="minor", length=0)
        ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    env = ChangingMaze()
    print(f"Maze before step {env.change_step}:")
    print(env.render(agent=False))

    # --- sanity checks -----------------------------------------------------
    assert env.next_cell(START, 0) == (4, 3)            # up into open cell
    assert env.next_cell((4, 3), 0) == (4, 3)           # up into wall -> stay
    assert env.next_cell((4, 8), 0) == (4, 8)           # shortcut still closed
    assert env.next_cell((4, 0), 0) == (3, 0)           # left gap is open
    assert env.next_cell(START, 1) == START             # off the bottom edge -> stay
    assert env.next_cell((0, 0), 2) == (0, 0)           # off the left edge -> stay

    env.state = (0, 7)
    s2, r, done = env.step(3)                           # right into G
    assert (s2, r, done) == (GOAL, 1.0, True) and env.state == START

    # run the clock to the change step and confirm the wall opens exactly then
    env.reset()
    for _ in range(CHANGE_STEP - 1):
        env.step(1)                                     # bump the bottom wall
    assert not env.changed and env.t == CHANGE_STEP - 1
    env.step(1)
    assert env.changed and env.t == CHANGE_STEP
    assert env.next_cell((4, 8), 0) == (3, 8)           # shortcut now open
    assert env.next_cell((4, 7), 0) == (4, 7)           # rest of the wall unchanged

    print(f"\nMaze from step {env.change_step}:")
    print(env.render(agent=False))
    print("\nAll environment checks passed.")
    print(f"Saved layout figure to {plot_layouts()}")
