import numpy as np


def bellman_backup(env, policy, V, state, gamma):
    """
    Bellman expectation equation for a single state:
        v(s) = sum_a pi(a|s) * sum_{s', r} p(s', r | s, a) * [r + gamma * v(s')]
    """
    value = 0.0
    for action in range(env.n_actions):
        for prob, next_state, reward in env.transitions(state, action):
            value += policy[state, action] * prob * (reward + gamma * V[next_state])
    return value


def iterative_policy_evaluation(env, policy, gamma=1.0, theta=1e-4, in_place=False, max_iters=10_000):
    """
    Iterative policy evaluation (Sutton & Barto, Section 4.1).

    Turns the Bellman expectation equation into an update rule and applies it
    to every non-terminal state, sweep after sweep, until the value function
    stops changing (largest change in a sweep < theta).

    in_place=False -> two-array version (default): each sweep computes V_{k+1}
                      only from V_k, so "change between successive iterations"
                      is exactly |V_{k+1}(s) - V_k(s)|
    in_place=True  -> overwrite V[s] immediately (uses newest values in the same sweep)

    Returns (V, iterations, history):
      V          - final value estimate for every state
      iterations - number of sweeps until convergence
      history[k] - V after sweep k (history[0] is the all-zero V_0)
    """
    # A. initialize all state values to zero (terminal states stay 0 forever)
    V = np.zeros(env.n_states)
    history = [V.copy()]
    iterations = 0

    while iterations < max_iters:
        iterations += 1
        delta = 0.0  # largest |V_new(s) - V_old(s)| seen in this sweep
        V_new = V if in_place else V.copy()

        # B. update every non-terminal state with the Bellman expectation backup
        for s in env.non_terminal_states:
            v_old = V[s]
            V_new[s] = bellman_backup(env, policy, V, s, gamma)
            delta = max(delta, abs(v_old - V_new[s]))

        V = V_new
        history.append(V.copy())

        # C. stop once the maximum change between successive iterations < theta
        if delta < theta:
            break

    return V, iterations, history


def solve_bellman_linear(env, policy, gamma=1.0):
    """
    Exact solution of the Bellman equations as a linear system.

    Over non-terminal states the Bellman equations are linear in v:
        v = r_pi + gamma * P_pi v   =>   (I - gamma * P_pi) v = r_pi
    where r_pi[s] is the expected one-step reward and P_pi[s, s'] the
    probability of moving s -> s' under pi. Terminal states have v = 0, so
    they simply drop out of P_pi.
    """
    non_terminal = [s for s in range(env.n_states) if not env.is_terminal(s)]
    index = {s: i for i, s in enumerate(non_terminal)}
    n = len(non_terminal)

    P = np.zeros((n, n))
    r = np.zeros(n)
    for s in non_terminal:
        i = index[s]
        for action in range(env.n_actions):
            for prob, next_state, reward in env.transitions(s, action):
                p = policy[s, action] * prob
                r[i] += p * reward
                if not env.is_terminal(next_state):
                    P[i, index[next_state]] += p

    v = np.linalg.solve(np.eye(n) - gamma * P, r)

    V = np.zeros(env.n_states)
    for s in non_terminal:
        V[s] = v[index[s]]
    return V


def print_value_table(V, size, decimals=2):
    # print V as a size x size grid, matching the gridworld layout
    for row in V.reshape(size, size):
        print("  " + " ".join(f"{v:>{decimals + 6}.{decimals}f}" for v in row))


if __name__ == "__main__":
    from gridworld import GridWorld, uniform_random_policy

    THETA = 1e-4
    env = GridWorld(size=4)
    policy = uniform_random_policy(env)

    V, iterations, _ = iterative_policy_evaluation(env, policy, gamma=1.0, theta=THETA)

    # D. display the number of iterations and the final value table
    print(f"Iterative policy evaluation (uniform random policy, gamma = 1, theta = {THETA:g})")
    print(f"Iterations until convergence: {iterations}\n")
    print("Final value table v_pi(s):")
    print_value_table(V, env.size)
