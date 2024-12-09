import gymnasium
import numpy as np
#lake_map_16 = ['SHFFHFFHFFFFFFFF', 'FFFHFHFFFFFFFFFH', 'FFFFFFFFFHFFFFHF', 'FHFFFHHFFFFFFFHF', 'FFFFFFFFFFHHFFFF', 'FFHFFFFFFFHHFFFF', 'FFHFFFFFFFFHHHFF', 'FFHFFFFFFFHHFFHF', 'HFHHFHFFHFFFFFFF', 'FFFFFFFFHFHHFFFF', 'FFFFFFFFFFFFFFFF', 'FFHFFHFFHHFFFFHF', 'HHFFHFFFFFFFFHFF', 'FHFFFFFFHFFFFFFF', 'FHHFFFFFFHFFFFFF', 'FFFHFFFHFFFHFFFG']

# env = gymnasium.make("FrozenLake-v1", is_slippery=False,  render_mode="human", map_name="4x4")
# q_table = np.load("q_table_qlearning_4.npy")
# q_table = np.load("q_table_sarsa_4.npy")


env = gymnasium.make("FrozenLake-v1", is_slippery=False,  render_mode="human", map_name="8x8")
q_table = np.load("q_table_qlearning_8.npy")
q_table = np.load("q_table_sarsa_8.npy")

lake_map = ['SFFFFFFF', 'FHFFFFHF', 'FFFFHFFF', 'FFFFFFFH', 'FFFFFFFF', 'FFFFFHFF', 'FFFFFFFF', 'FFFFFFFG']
env = gymnasium.make("FrozenLake-v1", is_slippery=False,render_mode="human", desc=lake_map)
q_table = np.load("q_table_qlearning_8.npy")

state, _ = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state]) 
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()

env.close()
