import torch.nn as nn
import torch.nn.functional as F

class KeanuModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3 * 64 * 64, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 2)

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        return out