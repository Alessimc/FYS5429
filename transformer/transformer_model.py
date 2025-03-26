import torch.nn as nn
from elan_block import ELAB

class ELAN(nn.Module):
    def __init__(self):
        super(ELAN, self).__init__()
        self.colors = 2  # Set to 2 for passive microwave data
        self.window_sizes = [4, 8, 16]
        self.m_elan  = 36
        self.c_elan  = 180
        self.n_share = 0
        self.r_expand = 2

        # Define head module
        m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # Define body module
        m_body = []
        for i in range(self.m_elan // (1+self.n_share)):
            if (i+1) % 2 == 1: 
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 0, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 1, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        
        # Define tail module (no PixelShuffle, output should match input size)
        m_tail = [nn.Conv2d(self.c_elan, self.colors, kernel_size=3, stride=1, padding=1)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]

        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)

        return x[:, :, :H, :W]  # Ensure output size matches input size

if __name__ == '__main__':
    pass