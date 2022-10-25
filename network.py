import torch.nn as nn
import torch.nn.functional as F
 
class QNetwork(nn.Module):
    
    
    def __init__(self, state_space: int, action_num: int, action_scale: int):
        super(QNetwork,self).__init__()
        # self.linear_1 = nn.Linear(state_space, state_space*s1)
        self.linear_1 = nn.Linear(state_space, 64)
        # self.linear_2 = nn.Linear(256, 64)
        
        # self.actions = [nn.Sequential(nn.Linear(state_space*s2, state_space*s3),
              # nn.LeakyReLU(leaky),
              # nn.Linear(state_space*s3, action_scale)
              # ) for _ in range(action_num)]

        # self.actions = nn.ModuleList(self.actions)

        # self.value = nn.Sequential(nn.Linear(state_space*s2, state_space*s3),
              # nn.LeakyReLU(leaky),
              # nn.Linear(state_space*s3, 1)
              # )
        # self.actions = [nn.Sequential(nn.Linear(state_space*s1, action_scale),
        self.actions = [nn.Sequential(nn.Linear(64, action_scale),
               nn.ReLU()
              ) for _ in range(action_num)]

        self.actions = nn.ModuleList(self.actions)

        # self.value = nn.Sequential(nn.Linear(state_space*s2, 1),
        self.value = nn.Sequential(nn.Linear(64, 1),
               nn.ReLU()
              )
        # self.double()      
              
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        # encoded = F.relu(self.linear_2(x))
        encoded = x
        actions = [x(encoded) for x in self.actions]
        value = self.value(encoded)
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1)[0].reshape(-1,1)
            actions[i] += value
        return actions
