import torch
import time
import torch.nn as nn

# BDQ
from utils import ReplayBuffer
from agent import BQN

num_cpu = 16
num_pf_per_core = 4
num_features_per_core = 7

# state_space = num_features_per_core*num_cpu
state_space = 176
action_space = num_cpu
action_scale = pow(2, num_pf_per_core)


total_reward = 0
alpha = 0.2
beta = 0.6
learning_rate = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

agent = BQN(state_space, action_space, action_scale,
            learning_rate, device, num_cpu, num_pf_per_core, alpha, beta)


model = agent.load_model("./models/model", device)
# model = agent.q
model = model.to(device)
print("Done")
'''
for name, param in model.named_parameters():
    if (name == "actions.223.0.weight"):
        print(torch.histc(param, 16))
        print(param)
'''
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def extract_model(model):
    new_state_dict = {}
    for name, weight in model['state_dict'].items():
        print(name, weight)
        # if 'backbone' in name:
            # new_state_dict[name] = weight
    
def run_model(model):
    duration = 0
    tot = 0
    ones = 0
    zeros = 0
    print("Model size ", count_parameters(model))
    tot_actions = [0] * num_cpu
    for i in range(100): 
        state = torch.rand((20, state_space)).float().to(device)
        tic = time.time()
        action = agent.action_test(state)
        for a in action:
            tot_actions[a]+=1
        toc = time.time()

        duration += (toc-tic)
       
    print("Duration", duration/100)
    print(tot_actions)
    

activation = {}


# for name, layer in model.named_modules():
     # model.register_forward_hook(getActivation(name))

     
def printnorm(self, input, output):
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('output norm:', output.data.norm())
    print("\n")


run_model(model)

model.linear_1.register_forward_hook(printnorm)
model.value[2].register_forward_hook(printnorm)

model.actions[0][2].register_forward_hook(printnorm)
model.actions[1][2].register_forward_hook(printnorm)
# model.actions[2][2].register_forward_hook(printnorm)
# model.actions[3][2].register_forward_hook(printnorm)
# model.actions[4][2].register_forward_hook(printnorm)
# model.actions[5][2].register_forward_hook(printnorm)
# model.actions[6][2].register_forward_hook(printnorm)

state = torch.rand((1, state_space)).float().to(device)
action = model(state.clone().detach())  

# print("------action-----------")
# print(action)
# print("----")

# print(model.linear_1.__dict__)
# print("-----------------")
# print(model.actions[0][2].__dict__)
print("-----------------")
# print(model.actions[1][2].__dict__)
# print("-----------------")
# print(model.actions[0][3].__dict__)
# print("-----------------")
# print(model.actions[0][4].__dict__)
# print("-----------------")


