import numpy as np
import gymnasium

env = gymnasium.make("FrozenLake-v1", is_slippery=False)
q_table = np.zeros((env.observation_space.n, env.action_space.n))

learning_rate = 0.01
gamma = 0.99  
epsilon = 1.0 
epsilon_min = 0.1  
epsilon_decay = 0.999  
num_episodes = 10000 

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + gamma * q_table[next_state,
                                     best_next_action] - q_table[state, action]
        )
        state = next_state

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 100 == 0:
        print(f"Episode {episode}: Total Reward = {
              total_reward}, Epsilon = {epsilon}")
        
np.save("q_table.npy", q_table)
