import torch.nn as nn
import torch.nn.init as init

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # added a layer 9-1-1-5 as in paper 
        self.conv1 = nn.Conv2d(2, 64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0) # new layer 
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 2, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        return x
