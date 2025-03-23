import torch
import torch.nn as nn

class DoubleIntegratorPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, action_range=[-1, 1]):
        super(DoubleIntegratorPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.action_range = action_range

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fcs = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.fc2 = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        for fc in self.fcs:
            x = torch.relu(fc(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=-1)
        return x

    def get_action(self, index):
        delta_action = (self.action_range[1] - self.action_range[0]) / self.action_dim
        return self.action_range[0] + delta_action * index + delta_action / 2