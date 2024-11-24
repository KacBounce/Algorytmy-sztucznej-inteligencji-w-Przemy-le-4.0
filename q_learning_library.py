import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    return gymnasium.make("FrozenLake-v1", is_slippery=False)
#env = DummyVecEnv([make_env])
env = gymnasium.make("FrozenLake-v1", is_slippery=False)

model = DQN("MlpPolicy", 
            env,
            verbose=1,
            learning_rate=0.01,  
            gamma=0.9,  
            exploration_initial_eps=1.0,  
            exploration_final_eps=0.1, 
            exploration_fraction=0.5,
            buffer_size=50000,
            batch_size=64)

model.learn(total_timesteps=10000)

model.save("dqn_frozenlake")
print("Model saved successfully!")

env.close()
