import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))

class CGNet(nn.Module):
    def __init__(self, num_classes=3, num_channels=7):
        super(CGNet, self).__init__()

        # Niveau 1 : Réduction initiale avec convolutions
        self.level1 = nn.Sequential(
            ConvBNReLU(num_channels, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(32, 32, kernel_size=3, stride=1, padding=1),
        )

        # Niveau 2 : Deux blocs simples avec réduction de résolution
        self.level2 = nn.Sequential(
            SimpleBlock(32, 64, stride=2),
            SimpleBlock(64, 64)
        )

        # Niveau 3 : Deux blocs simples pour extraire des caractéristiques plus profondes
        self.level3 = nn.Sequential(
            SimpleBlock(64, 128, stride=2),
            SimpleBlock(128, 128)
        )

        # Classifieur final
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.level1(x)  # Niveau 1
        x = self.level2(x)  # Niveau 2
        x = self.level3(x)  # Niveau 3
        x = self.classifier(x)  # Convolution de classification
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)  # Rétablir la taille originale
        return x