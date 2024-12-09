import numpy as np
import gymnasium

# Function to create FrozenLake environment with specific grid size
lake_map_16 = []

def create_frozenlake_env(size):
    """Creates a FrozenLake environment of the given size."""
    if size == 4:
        return gymnasium.make("FrozenLake-v1", is_slippery=False, map_name="4x4")
    elif size == 8:
        return gymnasium.make("FrozenLake-v1", is_slippery=False, map_name="8x8")
    else:
        # Generate a custom grid for sizes other than 4x4 or 8x8
        custom_map = generate_frozenlake_map(size)
        return gymnasium.make("FrozenLake-v1", is_slippery=False, desc=custom_map)

# Function to generate a custom map for arbitrary grid size


def generate_frozenlake_map(size):
    global lake_map_16
    """Generates a random FrozenLake map of the given size."""
    map_grid = []
    for i in range(size):
        row = []
        for j in range(size):
            if i == 0 and j == 0:
                row.append("S")  # Start point
            elif i == size - 1 and j == size - 1:
                row.append("G")  # Goal
            else:
                # Frozen or Hole
                row.append(np.random.choice(["F", "H"], p=[0.9, 0.1]))
        map_grid.append("".join(row))
    lake_map_16 = map_grid
    return map_grid

# Q-learning training function


def train_q_learning(env, num_episodes, learning_rate=0.01, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999):
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            best_next_action = np.argmax(q_table[next_state])
            q_table[state, action] += learning_rate * (
                reward + gamma * q_table[next_state,
                                         best_next_action] - q_table[state, action]
            )
            state = next_state

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, rewards

# Evaluation function


def evaluate_q_learning(q_table, env, num_episodes=100):
    total_rewards = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)

# Measure Q-table sparsity


def q_table_sparsity(q_table):
    num_zero_entries = np.sum(q_table == 0)
    total_entries = q_table.size
    return num_zero_entries / total_entries

# Test on different grid sizes


def test_on_grid_sizes(grid_sizes, num_train_episodes, num_eval_episodes):
    results = {}

    for size in grid_sizes:
        print(f"\nTraining on FrozenLake {size}x{size} grid...")
        env = create_frozenlake_env(size)

        # Train Q-learning
        q_table, rewards = train_q_learning(env, num_train_episodes)

        # Evaluate performance
        avg_reward = evaluate_q_learning(q_table, env, num_eval_episodes)

        # Calculate convergence episode
        avg_rewards_window = np.convolve(
            rewards, np.ones(100) / 100, mode="valid")
        convergence_episode = next(
            (i for i, r in enumerate(avg_rewards_window) if r >= 0.8), None)

        # Measure Q-table sparsity
        sparsity = q_table_sparsity(q_table)

        # Store results
        results[size] = {
            "Average Reward": avg_reward,
            "Convergence Episode": convergence_episode,
            "Q-Table Sparsity": sparsity,
        }

        print(f"Results for {size}x{size} grid:")
        print(f"  Average Reward: {avg_reward}")
        print(f"  Convergence Episode: {convergence_episode}")
        print(f"  Q-Table Sparsity: {sparsity:.2%}")
        
        if (size == 4):
            np.save(f"q_table_{size}.npy", q_table)
        elif(size == 8):
            np.save(f"q_table_{size}.npy", q_table)
        else:
           np.save(f"q_table_{size}.npy", q_table)
        
        #env.render()

    return results


# Run the test for 4x4, 8x8, and 16x16 grids
grid_sizes = [4, 8, 16]
num_train_episodes = 1000
num_eval_episodes = 100

results = test_on_grid_sizes(grid_sizes, num_train_episodes, num_eval_episodes)

print("\nFinal Comparison:")
for size, metrics in results.items():
    print(f"{size}x{size} Grid: {metrics}")

print(lake_map_16)
