import numpy as np
import gymnasium
import matplotlib.pyplot as plt
import time

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


# Global variables to track time
time_sarsa = 0
time_qlearning = 0

# Function to create FrozenLake environment with specific grid size


def create_frozenlake_env(size):
    """Creates a FrozenLake environment of the given size."""
    # Generate a custom grid for sizes other than 4x4 or 8x8
    custom_map = generate_frozenlake_map(size)
    return gymnasium.make("FrozenLake-v1", is_slippery=False, desc=custom_map)

# Function to generate a custom map for arbitrary grid size


def generate_frozenlake_map(size):
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
    map_file = open("maps.txt", "a")
    map_file.write(f"Size {size} : {map_grid}\n")
    map_file.close()
    return map_grid

# Q-learning training function


def train_q_learning(env, num_episodes, learning_rate=0.01, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99995):
    global time_qlearning
    time_start = time.time()
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    avg_rewards_window = np.zeros(num_episodes)
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

        # Moving average for learning curve
        avg_rewards_window[episode] = np.mean(
            rewards[max(0, episode-100):episode+1])

    time_end = time.time()
    time_qlearning = time_end - time_start
    return q_table, rewards, avg_rewards_window

# SARSA training function


def train_sarsa(env, num_episodes, learning_rate=0.01, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.99995):
    global time_sarsa
    time_start = time.time()
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    avg_rewards_window = np.zeros(num_episodes)
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        # Choose initial action using epsilon-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Choose next action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()  # Explore
            else:
                next_action = np.argmax(q_table[next_state])  # Exploit

            # Update Q-table using SARSA update rule
            q_table[state, action] += learning_rate * (
                reward + gamma * q_table[next_state,
                                         next_action] - q_table[state, action]
            )

            # Move to next state-action pair
            state, action = next_state, next_action

        rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Moving average for learning curve
        avg_rewards_window[episode] = np.mean(
            rewards[max(0, episode-100):episode+1])

    time_end = time.time()
    time_sarsa = time_end - time_start
    return q_table, rewards, avg_rewards_window

# Measure Q-table sparsity


def q_table_sparsity(q_table):
    num_zero_entries = np.sum(q_table == 0)
    total_entries = q_table.size
    return num_zero_entries / total_entries

# Test on different grid sizes for both Q-learning and SARSA


def test_on_grid_sizes(grid_sizes, num_train_episodes, num_eval_episodes):
    results = {}
    times_qlearning = []
    times_sarsa = []
    sparsities_qlearning = []
    sparsities_sarsa = []

    for size in grid_sizes:
        print(f"\nTraining on FrozenLake {size}x{size} grid...")

        env = create_frozenlake_env(size)

        # Train Q-learning
        q_table_qlearning, rewards_qlearning, avg_rewards_qlearning = train_q_learning(
            env, num_train_episodes)

        # Train SARSA
        q_table_sarsa, rewards_sarsa, avg_rewards_sarsa = train_sarsa(
            env, num_train_episodes)

        # Evaluate Q-learning
        avg_reward_qlearning = evaluate_q_learning(
            q_table_qlearning, env, num_eval_episodes)

        # Evaluate SARSA
        avg_reward_sarsa = evaluate_q_learning(
            q_table_sarsa, env, num_eval_episodes)

        # Measure Q-table sparsity
        sparsity_qlearning = q_table_sparsity(q_table_qlearning)
        sparsity_sarsa = q_table_sparsity(q_table_sarsa)

        # Track time and sparsity for graphs
        times_qlearning.append(time_qlearning)
        times_sarsa.append(time_sarsa)
        sparsities_qlearning.append(sparsity_qlearning)
        sparsities_sarsa.append(sparsity_sarsa)

        # Store results
        results[size] = {
            "Q-Learning": {
                "Average Reward": avg_reward_qlearning,
                "Q-Table Sparsity": sparsity_qlearning,
                "Learning Curve": avg_rewards_qlearning,
            },
            "SARSA": {
                "Average Reward": avg_reward_sarsa,
                "Q-Table Sparsity": sparsity_sarsa,
                "Learning Curve": avg_rewards_sarsa,
            }
        }

        print(f"Results for {size}x{size} grid:")
        print(f"  Q-Learning - Average Reward: {
              avg_reward_qlearning}, Sparsity: {sparsity_qlearning:.2%}")
        print(
            f"  SARSA - Average Reward: {avg_reward_sarsa}, Sparsity: {sparsity_sarsa:.2%}")

        # Save Q-tables for both algorithms
        np.save(f"q_table_qlearning_{size}.npy", q_table_qlearning)
        np.save(f"q_table_sarsa_{size}.npy", q_table_sarsa)

        # Plot learning curves
        plt.figure(figsize=(12, 6))
        plt.plot(avg_rewards_qlearning, label='Q-Learning', color='blue')
        plt.plot(avg_rewards_sarsa, label='SARSA', color='red')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title(f"Learning Curve for {size}x{size} Grid")
        plt.legend()
        plt.show()

    # Plot Time vs Grid Size
    plt.figure(figsize=(12, 6))
    plt.plot(grid_sizes, times_qlearning,
             label='Q-Learning', color='blue', marker='o')
    plt.plot(grid_sizes, times_sarsa, label='SARSA', color='red', marker='o')
    plt.xlabel('Grid Size')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time vs Grid Size')
    plt.legend()
    plt.show()

    # Plot Sparsity vs Grid Size
    plt.figure(figsize=(12, 6))
    plt.plot(grid_sizes, sparsities_qlearning,
             label='Q-Learning', color='blue', marker='o')
    plt.plot(grid_sizes, sparsities_sarsa,
             label='SARSA', color='red', marker='o')
    plt.xlabel('Grid Size')
    plt.ylabel('Q-table Sparsity')
    plt.title('Q-table Sparsity vs Grid Size')
    plt.legend()
    plt.show()

    return results


# Run the test for 4x4, 8x8, 16x16, and 32x32 grids
grid_sizes = [4, 6, 8]
num_train_episodes = 30000
num_eval_episodes = 100

results = test_on_grid_sizes(grid_sizes, num_train_episodes, num_eval_episodes)

# Final Comparison and Sparsity
print("\nFinal Comparison:")
for size, metrics in results.items():
    print(f"{size}x{size} Grid:")
    print(f"  Q-Learning: {metrics['Q-Learning']}")
    print(f"  SARSA: {metrics['SARSA']}")
print(f"S:{time_sarsa} Q:{time_qlearning}")
