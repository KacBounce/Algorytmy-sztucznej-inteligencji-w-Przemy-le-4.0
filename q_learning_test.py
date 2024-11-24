import gymnasium
import numpy as np

env = gymnasium.make("FrozenLake-v1", is_slippery=False,  render_mode="human")

q_table = np.load("q_table.npy")

state, _ = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state]) 
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()

env.close()
