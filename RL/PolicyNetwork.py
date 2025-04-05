import torch
import torch.nn as nn
from enum import Enum
import numpy as np

class ActionType(Enum):
    DISTRIBUTION=1,
    GAUSSIAN=2

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.training = False
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class DoubleIntegratorPolicy(Policy):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(DoubleIntegratorPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

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

    def get_action_dim(self):
        return self.action_dim
    
    def reset(self):
        pass

    def get_action_type(self):
        return ActionType.DISTRIBUTION

class DoubleIntegratorPolicyLSTM(Policy):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super(DoubleIntegratorPolicyLSTM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.lstm_out_dim = state_dim
        self.lstm = nn.LSTM(state_dim, self.lstm_out_dim, batch_first=True)
        self.reset()

        self.fc1 = nn.Linear(state_dim + self.lstm_out_dim, hidden_dims[0])
        self.fcs = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.fc2 = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        self.detach()

        lstm_in = x.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)

        lstm_out, (self.hidden, self.cell) = self.lstm(lstm_in, (self.hidden, self.cell))
        lstm_out = lstm_out.squeeze(0).squeeze(0)

        x_combined = torch.cat([x, lstm_out], dim=-1)

        x = torch.relu(self.fc1(x_combined))
        for fc in self.fcs:
            x = torch.relu(fc(x))
        x = self.fc2(x)
        x = torch.softmax(x, dim=-1)
        return x

    def get_action_dim(self):
        return self.action_dim
    
    def detach(self):
        self.hidden = self.hidden.detach()
        self.cell = self.cell.detach()
    
    def reset(self, batch_size=1):
        # Reset hidden state between episodes
        self.hidden = torch.zeros(1, batch_size, self.lstm_out_dim)
        self.cell = torch.zeros(1, batch_size, self.lstm_out_dim)

    def get_action_type(self):
        return ActionType.DISTRIBUTION

class GaussianPolicy(Policy):
    def __init__(self, state_dim, action_dim, hidden_dims, 
                 std_init=0.2, std_min=1e-5, std_max=0.6,
                 temperature_init=1, temperature_decay=1):
        
        super(GaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.temperature = temperature_init
        self.temperature_decay = temperature_decay

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fcs = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)

        self.std_init = std_init
        self.std_min = std_min
        self.std_max = std_max
        self.log_std_init = np.log(std_init)
        self.log_std_min = np.log(std_min)
        self.log_std_max = np.log(std_max)

        self.log_std = nn.Parameter(torch.ones(action_dim) * self.log_std_init, requires_grad=True)

    def forward(self, x):
        log_std = x
        x = torch.relu(self.fc1(x))
        for fc in self.fcs:
            x = torch.relu(fc(x))
        mean = torch.tanh(self.mean_head(x)) # Normalize the mean to [-1, 1] 

        log_std = self.log_std
        log_std = self.log_std_min + torch.nn.functional.softplus(log_std - self.log_std_min)
        log_std = self.log_std_max - torch.nn.functional.softplus(self.log_std_max - log_std)
        std = torch.exp(log_std) * self.temperature

        if self.training and self.temperature > 1e-9:
            self.temperature *= self.temperature_decay

        return mean, std
    
    def set_std(self, new_std):
        """
        Directly set the standard deviation (std) of the policy.
        Args:
            new_std (float or torch.Tensor): New standard deviation value(s).
        """
        if isinstance(new_std, float):
            new_std = torch.ones(self.action_dim) * new_std
        
        new_std = torch.clamp(new_std, self.std_min, self.std_max)
        
        # Convert std to log_std and update the parameter
        new_log_std = torch.log(new_std)
        with torch.no_grad():  # Avoid tracking this change in gradients
            self.log_std.copy_(new_log_std)

    def get_action_type(self):
        return ActionType.GAUSSIAN
    
    def reset(self):
        pass

class GaussianStateDependentPolicy(Policy):
    def __init__(self, state_dim, action_dim, hidden_dims, std_hidden_dim=16,
                 std_init=0.2, std_min=1e-5, std_max=0.6):
        super(GaussianStateDependentPolicy, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims

        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.fcs = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            self.fcs.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)

        self.std_init = std_init
        self.std_min = std_min
        self.std_max = std_max
        self.log_std_init = np.log(std_init)
        self.log_std_min = np.log(std_min)
        self.log_std_max = np.log(std_max)

        self.std_net_hidden_dim = std_hidden_dim
        self.std_net1 = nn.Linear(self.state_dim, self.std_net_hidden_dim)
        self.std_net2 = nn.Linear(self.std_net_hidden_dim, self.action_dim)
        nn.init.constant_(self.std_net2.bias, std_init)

    def forward(self, x):
        log_std = x
        x = torch.relu(self.fc1(x))
        for fc in self.fcs:
            x = torch.relu(fc(x))
        mean = torch.tanh(self.mean_head(x)) # Normalize the mean to [-1, 1] 

        log_std = self.std_net2(torch.relu(self.std_net1(log_std)))

        log_std = self.log_std_min + torch.nn.functional.softplus(log_std - self.log_std_min)
        log_std = self.log_std_max - torch.nn.functional.softplus(self.log_std_max - log_std)
        std = torch.exp(log_std)

        return mean, std

    def get_action_type(self):
        return ActionType.GAUSSIAN
    
    def reset(self):
        pass
    