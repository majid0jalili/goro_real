import torch
import time

# BDQ
from utils import ReplayBuffer
from agent import BQN
state_space = 44
action_space = 16
action_scale = 2
total_reward = 0

num_cpu = 4
num_pf_per_core = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory = ReplayBuffer(1000, action_space, device)
agent = BQN(state_space, action_space, action_scale,
            1e-4, device, num_cpu, num_pf_per_core)


agent.load_model("./models/model", device)

print("Done")

for i in range(100):
    state = torch.randint(0, 255, (1, 44)).float().to(device)
    action = agent.action(state)
    print(action)
