import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

# PVNet Model
class PVNet(nn.Module):
    def __init__(self, num_keypoints, num_classes):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes

        # Downsampling
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.res2 = self.make_layer(64, 128, blocks=2)
        self.res3 = self.make_layer(128, 256, blocks=2, stride=2)
        self.res4 = self.make_layer(256, 512, blocks=2, stride=1, dilation=2)  # bottleneck with dilation

        # Upsampling
        self.up1 = self.upsample_block(512, 256)
        self.up2 = self.upsample_block(512, 128)  # 256 (up1) + 256 (res3)
        self.up3 = self.upsample_block(256, 64)   # 128 (up2) + 128 (res2)

        # Output heads
        self.vector_head = nn.Conv2d(128, num_keypoints * 2, 1)  # 64 (up3) + 64 (initial)
        self.segmentation_head = nn.Conv2d(128, num_classes + 1, 1)

    def make_layer(self, in_c, out_c, blocks, stride=1, dilation=1):
        layers = [ResidualBlock(in_c, out_c, stride=stride, dilation=dilation)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c, dilation=dilation))
        return nn.Sequential(*layers)

    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        x1 = self.initial(x)  # [B, 64, H/4, W/4]
        x2 = self.res2(x1)    # [B, 128, H/4, W/4]
        x3 = self.res3(x2)    # [B, 256, H/8, W/8]
        x4 = self.res4(x3)    # [B, 512, H/8, W/8]

        u1 = self.up1(x4)               # [B, 256, H/4, W/4]
        u1 = torch.cat([u1, x3], dim=1) # [B, 512, H/4, W/4]

        u2 = self.up2(u1)               # [B, 128, H/2, W/2]
        u2 = torch.cat([u2, x2], dim=1) # [B, 256, H/2, W/2]

        u3 = self.up3(u2)               # [B, 64, H, W]
        u3 = torch.cat([u3, x1], dim=1) # [B, 128, H, W]

        vectors = self.vector_head(u3)        # [B, 2K, H, W]
        segmentation = self.segmentation_head(u3)  # [B, C+1, H, W]
        return vectors, segmentation
