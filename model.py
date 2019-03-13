import torch
from torch import nn

class simNN(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(simNN, self).__init__()
        hiddensize1 = 1024
        hiddensize2 = 512
        hiddensize3 = 256
        hiddensize4 = 128
        self.linear1 = nn.Linear(inchannel, hiddensize1)
        self.linear2 = nn.Linear(hiddensize1,hiddensize2)
        self.linear3 = nn.Linear(hiddensize2,hiddensize3)
        self.linear4 = nn.Linear(hiddensize3,outchannel)
        # self.linear5 = nn.Linear(hiddensize4, outchannel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)   # batch, 3*32*32
        out = self.relu(self.linear1(x))
        # torch.nn.Dropout(0.5)
        out = self.relu(self.linear2(out))
        out = self.relu(self.linear3(out))
        out = self.linear4(out) # batch, outchannel
        return out
