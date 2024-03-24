# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class CharTextCNN(nn.Module):
    def __init__(self,config):
        super(CharTextCNN,self).__init__()
        in_features = [config.char_num] + config.features[0:-1]
        out_features = config.features
        kernel_sizes = config.kernel_sizes
        self.convs = []
        self.conv1 = nn.Sequential(
                    nn.Conv1d(in_features[0], out_features[0], kernel_size=kernel_sizes[0], stride=1),
                    nn.BatchNorm1d(out_features[0]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=3, stride=3)
                )
        self.conv2  = nn.Sequential(
            nn.Conv1d(in_features[1], out_features[1], kernel_size=kernel_sizes[1], stride=1),
            nn.BatchNorm1d(out_features[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_features[2], out_features[2], kernel_size=kernel_sizes[2], stride=1),
            nn.BatchNorm1d(out_features[2]),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(in_features[3], out_features[3], kernel_size=kernel_sizes[3], stride=1),
            nn.BatchNorm1d(out_features[3]),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(in_features[4], out_features[4], kernel_size=kernel_sizes[4], stride=1),
            nn.BatchNorm1d(out_features[4]),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(in_features[5], out_features[5], kernel_size=kernel_sizes[5], stride=1),
            nn.BatchNorm1d(out_features[5]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=config.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=config.dropout)
        )

        self.fc3 = nn.Linear(1024, config.num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x