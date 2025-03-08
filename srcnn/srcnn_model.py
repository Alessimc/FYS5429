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

        # Initialize weights with Gaussian (mean=0, std=0.001) -> worse results than default
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


class SRCNN2(nn.Module):
    """
    A wider model than SRCNN
    """
    def __init__(self):
        super(SRCNN2, self).__init__()
        # increased filters
        self.conv1 = nn.Conv2d(2, 128, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0) # wider (more filters)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 2, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x
    
class SRCNN3(nn.Module):
    """
    A deeper model than SRCNN
    """
    def __init__(self):
        super(SRCNN3, self).__init__()
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

class SRCNN4(nn.Module):
    """
    A wider and deeper model combining the strengths of SRCNN2 and SRCNN3
    """
    def __init__(self):
        super(SRCNN4, self).__init__()
        # Increase the width (number of filters)
        self.conv1 = nn.Conv2d(2, 128, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)  # Wider than SRCNN3
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 2, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        return x
    
class SRCNN5(nn.Module):
    """
    An even wider and deeper SRCNN model
    """
    def __init__(self):
        super(SRCNN5, self).__init__()
        # Wider first layer
        self.conv1 = nn.Conv2d(2, 256, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        # More layers (deeper than SRCNN4)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Slightly larger kernel
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(64, 2, kernel_size=5, stride=1, padding=2)  # Final layer

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x


class SRCNN6(nn.Module):
    """
    A wider model than SRCNN2
    """
    def __init__(self):
        super(SRCNN6, self).__init__()
        # Increased filters even more
        self.conv1 = nn.Conv2d(2, 256, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 2, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x