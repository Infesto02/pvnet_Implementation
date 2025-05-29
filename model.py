import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class PVNet(nn.Module):
    def __init__(self, num_keypoints, num_classes):
        super(PVNet, self).__init__()
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res2 = self._make_layer(64, 128, num_blocks=2, stride=1)
        self.res3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.res4 = self._make_layer(256, 512, num_blocks=2, stride=1, dilation=2)  # bottleneck with dilation

        self.up1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(256 + 256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up3 = nn.Sequential(
            nn.Conv2d(128 + 128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.vector_head = nn.Conv2d(64 + 64, num_keypoints * 2, kernel_size=1)
        self.segmentation_head = nn.Conv2d(64 + 64, num_classes + 1, kernel_size=1)  # +1 for background

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dilation=1):
        layers = [ResidualBlock(in_channels, out_channels, stride=stride, dilation=dilation)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)      # [B, 64, H/4, W/4]
        x2 = self.res2(x1)       # [B, 128, H/4, W/4]
        x3 = self.res3(x2)       # [B, 256, H/8, W/8]
        x4 = self.res4(x3)       # [B, 512, H/8, W/8] (dilated conv)

        u1 = self.up1(x4)        # [B, 256, H/4, W/4]
        u1 = torch.cat([u1, x3], dim=1)
        u2 = self.up2(u1)        # [B, 128, H/2, W/2]
        u2 = torch.cat([u2, x2], dim=1)
        u3 = self.up3(u2)        # [B, 64, H, W]
        u3 = torch.cat([u3, x1], dim=1)

        vectors = self.vector_head(u3)
        segmentation = self.segmentation_head(u3)
        return vectors, segmentation
