from typing import Tuple
import torch.nn as nn

from .blocks_unet import ConvBlock, DownBlock, UpBlock, FinalBlock


class Unet2D_fancy(nn.Module):
    def __init__(self, channels: Tuple[int], num_class: int) -> None:
        super(Unet2D_fancy, self).__init__()
        self.channels = channels
        self.num_class = num_class
        # channels = (3, 64, 128, 256, 512, 1024)
        self.numC = len(channels)
        self.input_conv = ConvBlock(channels[0], channels[1])
        self.enc_blocks = nn.ModuleList(
            [DownBlock(channels[i], channels[i+1]) for i in range(1, self.numC-1)])
        self.dec_blocks = nn.ModuleList(
            [UpBlock(channels[i], channels[i-1]) for i in range(self.numC-1, 0, -1)])
        self.output_conv = FinalBlock(channels[1], num_class)

    def forward(self, x):
        # Encoder Part
        x1 = self.input_conv(x)
        features = [x1]
        xtmp = x1
        for e_blk in self.enc_blocks:
            xtmp = e_blk(xtmp)
            features.append(xtmp)
        # Decoder Part: Go reverse along the collected features
        xtmp = features[-1]  # initial value (encoder's last feature)
        for i, d_blk in enumerate(self.dec_blocks):
            idx = len(features) - 1 - i
            xtmp = d_blk(xtmp, features[idx-1])  # note idx - 1
        logits = self.output_conv(xtmp)
        return logits


class Unet2D_simple(nn.Module):

    def __init__(self, channels: int, num_class: int) -> None:
        super(Unet2D_simple, self).__init__()
        self.channels = channels
        self.num_class = num_class
        # channels = (1, 64, 128, 256, 512, 1024)
        self.input_conv = ConvBlock(channels, 64)
        self.output_conv = FinalBlock(64, num_class)

        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        # self.down3 = DownBlock(256, 512)
        # self.down4 = DownBlock(512, 1024)
        # self.up1 = UpBlock(1024, 512)
        # self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # x5 = self.down4(x4)
        # x = self.up1(x5, x4)
        # x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.output_conv(x)
        return logits

class Unet2D_org(nn.Module):

    def __init__(self, channels: int, num_class: int) -> None:
        super(Unet2D_org, self).__init__()
        self.channels = channels
        self.num_class = num_class
        # channels = (1, 64, 128, 256, 512, 1024)
        self.input_conv = ConvBlock(channels, 64)
        self.output_conv = FinalBlock(64, num_class)

        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)

    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.output_conv(x)
        return logits
