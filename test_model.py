import torch
import time

# BDQ
from utils import ReplayBuffer
from agent import BQN
num_cpu = 16
num_pf_per_core = 4
num_features_per_core = 6

state_space = num_features_per_core*num_cpu
action_space = num_pf_per_core*num_cpu
action_scale = 2
total_reward = 0
alpha = 0.2
beta = 0.6
learning_rate = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory = ReplayBuffer(1000, action_space, device)
agent = BQN(state_space, action_space, action_scale,
            learning_rate, device, num_cpu, num_pf_per_core, alpha, beta)


model = agent.load_model("./models/model_raw", device)

print("Done")
'''
for name, param in model.named_parameters():
    if (name == "actions.223.0.weight"):
        print(torch.histc(param, 16))
        print(param)
'''
for i in range(100):
    state = torch.randint(0, 255, (1, state_space)).float().to(device)
    action = agent.action(state)
    ones = sum(x.count(1) for x in action)
    zeros = sum(x.count(0) for x in action)
    tot = ones + zeros
    print("Fraction 0  1", zeros/tot, ones/tot)
