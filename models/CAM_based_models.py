import torch
import torch.nn as nn
import torch.nn.functional as F

class MTEXCNN(nn.Module):
    def __init__(self, time_length, n_classes):
        super(MTEXCNN, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, (time_length//2 + 1, 1))
        self.conv_2 = nn.Conv2d(16, 32, (time_length//4 + 1, 1))
        self.conv_3 = nn.Conv2d(32, 1, 1)
        self.conv_4 = nn.Conv1d(3, 64, 3)
        self.linear_1 = nn.Linear(64 * (time_length//4 + 1 - 1), 32)
        self.linear_2 = nn.Linear(32, n_classes)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.conv_3(x)
        x = F.relu(x)
        x = x.view(-1,x.size(3),x.size(2))
        x = self.conv_4(x)
        x = F.relu(x)
        x = x.view(x.size(0),-1)
        x = self.linear_1(x)
        x = self.linear_2(x)