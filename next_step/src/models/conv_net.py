import torch.nn as nn

class SimpleConvNet(nn.Module):
    def __init__(self, config):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, config['model']['conv1_out'], 3)
        self.conv2 = nn.Conv2d(config['model']['conv1_out'], config['model']['conv2_out'], 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, config['model']['fc1_out'])
        self.fc2 = nn.Linear(config['model']['fc1_out'], 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 1600)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x