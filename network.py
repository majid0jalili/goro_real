import torch.nn as nn
import torch
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self, state_space: int, action_space: int, action_scale: int):
        super(QNetwork, self).__init__()
 
        self.linear_1 = nn.Linear(state_space, 256)
        self.linear_2 = nn.Linear(256, 256)
        
        self.actions = [nn.Sequential(nn.Linear(256, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, action_scale)
                                      ) for _ in range(action_space)]

        
        
        self.actions = nn.ModuleList(self.actions)
        
        self.value = nn.Sequential(nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 1)
                                   )
        print("self.linear_1", self.linear_1)       

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = torch.relu(self.linear_2(x))
        encoded = x 
        actions = [x(encoded) for x in self.actions]
        value = self.value(encoded)
        
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1, 1)
            actions[i] += value
        return actions
