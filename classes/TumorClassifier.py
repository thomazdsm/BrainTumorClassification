import torch
import torch.nn as nn


class TumorClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(TumorClassifier, self).__init__()
        self.model = nn.Sequential(
            self._conv_block(1, 32),
            self._conv_block(32, 32),
            self._conv_block(32, 64),
            self._conv_block(64, 128),
            nn.Flatten(),
            nn.Linear(25088, num_classes)
        )

    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
