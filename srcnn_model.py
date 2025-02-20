import torch.nn as nn
import torch.nn.init as init

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 2, kernel_size=5, stride=1, padding=2)

        # Initialize weights with Gaussian (mean=0, std=0.001)
        # self._initialize_weights()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    # def _initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             init.normal_(m.weight, mean=0, std=0.001)  # Gaussian with std=0.001
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)  # Bias set to 0