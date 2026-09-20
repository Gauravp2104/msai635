# Assignment 1 — K-Armed Bandit

Implementation of the k-armed bandit testbed and greedy / epsilon-greedy
action-value agents, per the assignment spec in
`K-Armed Bandit Problem Assignment-3.pdf`.

## Files

- `bandit.py` — `KArmedBandit`: a k-armed bandit with true action values
  drawn from `N(0, 1)` and rewards drawn from `N(q*(a), 1)`.
- `agents.py` — `GreedyAgent` and `EpsilonGreedyAgent`, using sample-average
  action-value estimates.
- `experiment.py` — runs the greedy vs. epsilon-greedy comparison over
  multiple runs/steps, prints a results table, and saves a performance plot.

## Setup

Requires Python 3 with `numpy` and `matplotlib`.

```bash
cd assignment1
python3 -m venv venv
source venv/bin/activate
pip install numpy matplotlib
```

(A `venv/` may already exist in this directory — if so, just `source venv/bin/activate`.)

## Running

Sanity-check a single bandit instance (prints true action values and one
reward sample per arm):

```bash
python3 bandit.py
```

Run the full experiment (200 runs x 1000 steps, comparing greedy and
epsilon-greedy agents with epsilon = 0.01 and 0.1):

```bash
python3 experiment.py
```

This prints an average-reward / % optimal-action table at checkpoint steps
for each agent, and saves a timestamped plot,
`epsilon_greedy_agent_performance_<YYYYMMDD_HHMMSS>.png`, comparing average
reward and % optimal action over time.

To change the experiment parameters (steps, runs, epsilon values), edit the
`STEPS`, `RUNS`, and `EPSILONS` constants at the top of the `__main__` block
in `experiment.py`.
