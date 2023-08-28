import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class QNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_units: int, output_dim: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_dim)
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)
        self.dropout = nn.Dropout(p=0.2)

        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -0.01, 0.01))

        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x: torch.Tensor):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def get_params(self):
        return [param.clone().detach() for param in self.parameters()]

    def set_params(self, params):
        for param, new_param in zip(self.parameters(), params):
            param.data.copy_(new_param)
