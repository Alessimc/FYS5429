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

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# different kernels
class SRCNN_K3(nn.Module):
    def __init__(self):
        super(SRCNN_K3, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # Changed kernel size to 3
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 2, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

class SRCNN_K5(nn.Module):
    def __init__(self):
        super(SRCNN_K5, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=9, stride=1, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2)  # Changed kernel size to 5
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 2, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

# deeper
class SRCNN_L4(nn.Module):
    """
    A deeper model than SRCNN
    """
    def __init__(self):
        super(SRCNN_L4, self).__init__()
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

# wider
class SRCNN_W128(nn.Module):
    """
    A wider model than SRCNN
    """
    def __init__(self):
        super(SRCNN_W128, self).__init__()
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
    



class SRCNN_W256(nn.Module):
    """
    A wider model than SRCNN_W128
    """
    def __init__(self):
        super(SRCNN_W256, self).__init__()
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