import numpy as np

from gridworld import ACTION_NAMES

ACTION_ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def q_value(env, V, state, action, gamma):
    """
    One-step lookahead value of taking `action` in `state`, then following V:
        q(s, a) = sum_{s', r} p(s', r | s, a) * [r + gamma * V(s')]
    """
    return sum(prob * (reward + gamma * V[next_state])
               for prob, next_state, reward in env.transitions(state, action))


def bellman_optimality_backup(env, V, state, gamma):
    """
    Bellman optimality equation for a single state:
        v*(s) = max_a sum_{s', r} p(s', r | s, a) * [r + gamma * v*(s')]
    Same as the expectation backup in policy_evaluation.py, except the
    average over pi(a|s) is replaced by a max over actions.
    """
    return max(q_value(env, V, state, a, gamma) for a in range(env.n_actions))


def value_iteration(env, gamma=1.0, theta=1e-4, in_place=False, max_iters=10_000):
    """
    Value iteration (Sutton & Barto, Section 4.4).

    Starts from V = 0 and applies the Bellman optimality backup to every
    non-terminal state until the largest change in a sweep is < theta.

    in_place=False -> two-array version (default): V_{k+1} built only from V_k
    in_place=True  -> overwrite V[s] immediately within the sweep

    Returns (V, iterations, history):
      V          - optimal value function v*
      iterations - number of sweeps until convergence
      history[k] - V after sweep k (history[0] is the all-zero V_0)
    """
    # initialize all state values to zero (terminal states stay 0 forever)
    V = np.zeros(env.n_states)
    history = [V.copy()]
    iterations = 0

    while iterations < max_iters:
        iterations += 1
        delta = 0.0  # largest |V_new(s) - V_old(s)| seen in this sweep
        V_new = V if in_place else V.copy()

        # update every non-terminal state with the Bellman optimality backup
        for s in env.non_terminal_states:
            v_old = V[s]
            V_new[s] = bellman_optimality_backup(env, V, s, gamma)
            delta = max(delta, abs(v_old - V_new[s]))

        V = V_new
        history.append(V.copy())

        # stop once the maximum change between successive iterations < theta
        if delta < theta:
            break

    return V, iterations, history


def extract_greedy_policy(env, V, gamma=1.0, tol=1e-6):
    """
    Derive the optimal policy from v*: in each state pick the action(s) that
    maximize the one-step lookahead q(s, a). Ties are kept (several actions can
    be equally optimal), and pi*(a|s) splits probability evenly among them.

    Returns (policy, best_actions):
      policy       - (n_states, n_actions) array, pi*(a|s)
      best_actions - list where best_actions[s] is the list of optimal actions
                     (empty for terminal states)
    """
    policy = np.zeros((env.n_states, env.n_actions))
    best_actions = [[] for _ in range(env.n_states)]

    for s in env.non_terminal_states:
        q = np.array([q_value(env, V, s, a, gamma) for a in range(env.n_actions)])
        # every action within tol of the best q-value counts as optimal
        best = np.flatnonzero(np.isclose(q, q.max(), atol=tol))
        best_actions[s] = best.tolist()
        policy[s, best] = 1.0 / len(best)

    return policy, best_actions


def print_value_table(V, size, decimals=2):
    # print V as a size x size grid, matching the gridworld layout
    for row in V.reshape(size, size):
        print("  " + " ".join(f"{v:>{decimals + 6}.{decimals}f}" for v in row))


def print_policy_arrows(env, best_actions):
    # one cell per state: the arrows of all optimal actions, "T" for terminal
    for row in range(env.size):
        cells = []
        for col in range(env.size):
            s = env.to_state(row, col)
            if env.is_terminal(s):
                cells.append("T")
            else:
                cells.append("".join(ACTION_ARROWS[a] for a in best_actions[s]))
        print("  " + " ".join(f"{c:^6}" for c in cells))


def print_policy_labels(env, best_actions):
    # text-label version of the same policy, one state per line
    for s in range(env.n_states):
        if env.is_terminal(s):
            label = "terminal"
        else:
            label = ", ".join(ACTION_NAMES[a] for a in best_actions[s])
        print(f"  s{s:<2}: {label}")


if __name__ == "__main__":
    from gridworld import GridWorld

    GAMMA = 1.0
    THETA = 1e-4
    env = GridWorld(size=4)

    V_star, iterations, history = value_iteration(env, gamma=GAMMA, theta=THETA)

    print(f"Value iteration (gamma = {GAMMA:g}, theta = {THETA:g})")
    print(f"Iterations until convergence: {iterations}\n")

    # show how v* is built up sweep by sweep
    for k in range(len(history)):
        print(f"V_k after k = {k} sweeps:")
        print_value_table(history[k], env.size)
        print()

    # a) optimal value function
    print("a) Optimal value function v*(s):")
    print_value_table(V_star, env.size)
    print()

    # b) optimal policy, greedy with respect to v*
    _, best_actions = extract_greedy_policy(env, V_star, gamma=GAMMA)

    # c) policy shown with arrows and with text labels
    print("b/c) Optimal policy pi*(s) (arrows; several arrows = tied optimal actions, T = terminal):")
    print_policy_arrows(env, best_actions)
    print()
    print("Optimal policy as text labels:")
    print_policy_labels(env, best_actions)
