from mimetypes import init
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as nnFunc


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.double_conv_block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv_block(x)
        return x


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(input_channels, output_channels)
        )

    def forward(self, x):
        x = self.down_block(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, bilinear=False) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            input_channels, input_channels//2, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(input_channels, output_channels)

    def forward(self, x1, x2):
        # x1 or x2 shape is (N,C,H,W)
        x1 = self.upsample(x1)
        deltaX = x2.size()[-2] - x1.size()[-2]
        deltaY = x2.size()[-1] - x1.size()[-1]
        x1 = nnFunc.pad(x1, (deltaX//2, deltaX - deltaX //
                   2, deltaY//2, deltaY - deltaY//2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_block(x)
        return x


class FinalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        # A simple final conv layer
        self.layer = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x):
        x = self.layer(x)
        return x
