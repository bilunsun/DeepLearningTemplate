import torch
from torch import nn


class ExampleMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(28**2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), 28**2)
        return self.layers(x)
