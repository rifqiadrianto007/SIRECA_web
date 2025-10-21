import torch
import torch.nn as nn

class SmoothCNN(nn.Module):
    def __init__(self):
        super(SmoothCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),  # <== ubah 64 jadi 32
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)
