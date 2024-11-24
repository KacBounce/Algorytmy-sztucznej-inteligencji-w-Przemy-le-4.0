# test_dqn.py
import gymnasium
from stable_baselines3 import DQN

env = gymnasium.make("FrozenLake-v1", is_slippery=False, render_mode="human")

model = DQN.load("dqn_frozenlake")
print("Model loaded successfully!")

state, _ = env.reset()
done = False

print("Testing trained agent:")
while not done:
    action, _ = model.predict(state)
    action = int(action)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    env.render()

env.close()
