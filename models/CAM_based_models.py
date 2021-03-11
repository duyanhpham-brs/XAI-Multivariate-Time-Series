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
        self.fc1 = nn.Linear(64 * (time_length//4), 32)
        self.fc2 = nn.Linear(32, n_classes)
        
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
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class XCM(nn.Module):
    def __init__(self, window_size, time_length, feature_length, n_classes):
        super(XCM, self).__init__()
        self.conv_11 = nn.Conv2d(1, 16, window_size, padding=window_size//2)
        self.batchnorm_11 = nn.BatchNorm2d(16)
        self.conv_12 = nn.Conv2d(16, 1, 1)
        self.conv_21 = nn.Conv1d(3, 16, window_size, padding=window_size//2)
        self.batchnorm_21 = nn.BatchNorm1d(16)
        self.conv_22 = nn.Conv1d(16, 1, 1)
        self.conv_3 = nn.Conv1d(time_length, 32, window_size, padding=window_size//2)
        self.batchnorm_3 = nn.BatchNorm1d(32)
        self.glb_avg_pool = nn.AvgPool1d(feature_length + 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, n_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # 2d (spatial) branch
        first_branch1 = self.conv_11(x)
        first_branch1 = self.batchnorm_11(first_branch1)
        first_branch1 = F.relu(first_branch1)
        first_branch2 = self.conv_12(first_branch1)
        first_branch2 = F.relu(first_branch2)
        # 1d (temporal) branch
        second_branch1 = x.view(-1,x.size(3),x.size(2))
        second_branch1 = self.conv_21(second_branch1)
        second_branch1 = self.batchnorm_21(second_branch1)
        second_branch1 = F.relu(second_branch1)
        second_branch2 = self.conv_22(second_branch1)
        second_branch2 = F.relu(second_branch2)
        # Concatenation
        first_branch2 = first_branch2.view(-1, first_branch2.size(3), first_branch2.size(2))
        main_branch = torch.cat((first_branch2, second_branch2), 1)
        main_branch = main_branch.view(-1, main_branch.size(2), main_branch.size(1))
        main_branch = self.conv_3(main_branch)
        main_branch = self.batchnorm_3(main_branch)
        main_branch = F.relu(main_branch)
        main_branch = self.glb_avg_pool(main_branch)
        main_branch = self.flatten(main_branch)
        main_branch = self.fc1(main_branch)
        main_branch = self.softmax(main_branch)

        return main_branch
