import torch
import torch.nn as nn


class SimpleNetModel(nn.Module):

    def __init__(self, hidden=10, output=1):
        super().__init__()
        self.net1 = nn.Linear(hidden, 5)
        self.act1 = nn.relu()
        self.net2 = nn.Linear(5, output)

    def forward(self, x):

        x = self.net1(x)
        x = self.act1(x)
        x = self.net2(x)
        return x
