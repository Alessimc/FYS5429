import torch.nn as nn
import torch
try:
    from elan_block import ELAB
except:
    from .elan_block import ELAB

class ELAN(nn.Module):
    def __init__(self):
        super(ELAN, self).__init__()
        self.colors = 2  # Set to 2 for passive microwave data
        self.window_sizes = [2, 4, 8]
        self.m_elan  = 24
        self.c_elan  = 60
        self.n_share = 1
        self.r_expand = 1

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

    # def forward(self, x):
    #     H, W = x.shape[2:]

    #     x = self.head(x)
    #     res = self.body(x)
    #     res = res + x
    #     x = self.tail(res)

    #     return x[:, :, :H, :W]  # Ensure output size matches input size
    
    # def forward(self, x):
    #     H, W = x.shape[2:]

    #     x = self.head(x)
    #     res = x
        
    #     # Collect the attention maps from all the ELAB modules in the body
    #     attention_maps = []
    #     for i, layer in enumerate(self.body):
    #         res, atn = layer(res)  # Each ELAB layer returns both output and attention maps
    #         attention_maps.append(atn)  # Store the attention maps for later use

    #     res = res + x  # Add the input to the residual
    #     x = self.tail(res)

    #     # Compute the final attention map (average of all attention maps)
    #     # final_attention_map = torch.mean(torch.stack(attention_maps), dim=0)

    #     return x[:, :, :H, :W], #final_attention_map  # Return both output and final attention map

    def forward(self, x, return_attention=False):
        H, W = x.shape[2:]

        x = self.head(x)
        res = x
        
        attention_maps = []
        for i, layer in enumerate(self.body):
            res, atn = layer(res)
            attention_maps.append(atn)

        res = res + x
        x = self.tail(res)

        if return_attention:
            # Flatten list of lists
            final_attention_map = attention_maps[-1]
            return x[:, :, :H, :W], final_attention_map
        else:
            return x[:, :, :H, :W]

if __name__ == '__main__':
    pass