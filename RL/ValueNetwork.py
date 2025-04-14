import torch
import torch.nn as nn
import numpy as np

class SimpleValueFunction(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super(SimpleValueFunction, self).__init__()
        self.state_dim = state_dim
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fcs = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.fc2 = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        # x may contain actions information used by policy network.
        # Value network does not need this information.
        x = x[0:self.state_dim]
        x = torch.relu(self.fc1(x))
        for fc in self.fcs:
            x = torch.relu(fc(x))
        x = self.fc2(x)
        return x

class SimpleQFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(SimpleQFunction, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.fcs = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.fc2 = nn.Linear(hidden_dims[-1], 1)

        self._initialize_weights()

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = torch.relu(self.fc1(x))
        for fc in self.fcs:
            x = torch.relu(fc(x))
        x = self.fc2(x)
        return x.squeeze(-1)
    
    def _initialize_weights(self):
        # Orthogonal initialization (better for RL)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        for layer in self.fcs:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
            
        nn.init.orthogonal_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc2.bias, 0.0)
