import random
from collections import Counter

# ---------------------------------------------------------------------------
# Starter code: MDP, policies and episode generation
# ---------------------------------------------------------------------------
states = ["A", "B"]

# behavior policy b: uniform over actions
behavior_policy = {
    "A": {"Left": 0.5, "Right": 0.5},
    "B": {"Left": 0.5, "Right": 0.5},
}

# target policy pi: always chooses Left
target_policy = {
    "A": {"Left": 1.0, "Right": 0.0},
    "B": {"Left": 1.0, "Right": 0.0},
}


def generate_episode():
    """
    Run one episode from A following the behavior policy.
    Returns a list of (state, action, reward) tuples, one per time step.

    Transitions:
      A: Left  -> B         reward 0
         Right -> Terminal  reward 1
      B: Left  -> Terminal  reward 2
         Right -> Terminal  reward 0
    """
    episode = []
    state = "A"
    while state != "Terminal":
        probs = behavior_policy[state]
        action = random.choices(
            list(probs.keys()),
            weights=list(probs.values())
        )[0]

        if state == "A":
            if action == "Left":
                next_state = "B"
                reward = 0
            else:
                next_state = "Terminal"
                reward = 1
        else:
            if action == "Left":
                next_state = "Terminal"
                reward = 2
            else:
                next_state = "Terminal"
                reward = 0

        episode.append((state, action, reward))
        state = next_state
    return episode


# ---------------------------------------------------------------------------
# Part 1: generate 10,000 episodes
# ---------------------------------------------------------------------------
N_EPISODES = 10_000
SEED = 0


def generate_episodes(n_episodes=N_EPISODES):
    return [generate_episode() for _ in range(n_episodes)]


def episode_to_str(episode):
    return " -> ".join(f"{s}({a}, r={r})" for s, a, r in episode) + " -> Terminal"


if __name__ == "__main__":
    random.seed(SEED)

    episodes = generate_episodes(N_EPISODES)
    print(f"Generated {len(episodes)} episodes with the behavior policy b (seed={SEED})\n")

    print("First 5 episodes:")
    for i, ep in enumerate(episodes[:5]):
        print(f"  {i}: {episode_to_str(ep)}")

    # the MDP has only three possible trajectories; count how often each appears
    counts = Counter(tuple((s, a) for s, a, _ in ep) for ep in episodes)
    print("\nTrajectory frequencies (expected under b: A-Right 0.50, A-Left,B-Left 0.25, A-Left,B-Right 0.25):")
    for traj, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        label = ", ".join(f"{s}-{a}" for s, a in traj)
        print(f"  {label:<22} {c:>6}  ({c / N_EPISODES:.4f})")

    lengths = Counter(len(ep) for ep in episodes)
    print(f"\nEpisode lengths: {dict(sorted(lengths.items()))}")
