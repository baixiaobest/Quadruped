import torch
import torch.nn as nn

class SimpleValuePolicy(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super(SimpleValuePolicy, self).__init__()
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
