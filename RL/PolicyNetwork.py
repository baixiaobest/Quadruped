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
    
    def reset(self):
        pass

class DoubleIntegratorPolicyLSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims, action_range=[-1, 1]):
        super(DoubleIntegratorPolicyLSTM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.action_range = action_range

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

    def get_action(self, index):
        delta_action = (self.action_range[1] - self.action_range[0]) / self.action_dim
        return self.action_range[0] + delta_action * index + delta_action / 2
    
    def detach(self):
        self.hidden = self.hidden.detach()
        self.cell = self.cell.detach()
    
    def reset(self, batch_size=1):
        # Reset hidden state between episodes
        self.hidden = torch.zeros(1, batch_size, self.lstm_out_dim)
        self.cell = torch.zeros(1, batch_size, self.lstm_out_dim)