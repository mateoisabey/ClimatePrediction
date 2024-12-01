import torch
import torch.nn as nn
import torch.nn.functional as F

class Wrap(nn.Module):
    def __init__(self, padding):
        super(Wrap, self).__init__()
        self.p = padding

    def forward(self, x):
        if self.p > 0:
            x = torch.cat([x[:, :, :, -self.p:], x, x[:, :, :, :self.p]], dim=3)  # Padding circulaire sur la largeur
            x = torch.cat([x[:, :, -self.p:, :], x, x[:, :, :self.p, :]], dim=2)  # Padding circulaire sur la hauteur
        return x

class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.padding(input)
        output = self.conv(output)
        output = self.bn(output)
        output = self.act(output)
        return output

class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, groups=nIn, bias=False)

    def forward(self, input):
        output = self.padding(input)
        output = self.conv(output)
        return output

class SqueezeExciteBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SqueezeExciteBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)
        self.F_loc = ChannelWiseConv(n, n, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(n, n, 3, 1, d=dilation_rate)
        self.bn_prelu = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)
        self.add = add
        self.attention = SqueezeExciteBlock(nOut, reduction)

        self.match_dims = None
        if nIn != nOut:
            self.match_dims = nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn_prelu(joi_feat)
        joi_feat = self.act(joi_feat)
        output = self.attention(joi_feat)

        if self.add:
            if self.match_dims is not None:
                input = self.match_dims(input)
            output = input + output
        return output

class CGNet(nn.Module):
    def __init__(self, num_classes=3, num_channels=16):
        super(CGNet, self).__init__()
        self.level1 = nn.Sequential(
            ConvBNPReLU(num_channels, 32, 3, 2),
            ConvBNPReLU(32, 32, 3, 1),
            ConvBNPReLU(32, 32, 3, 1),
        )

        self.level2_0 = ContextGuidedBlock(32, 64, dilation_rate=2, reduction=16)
        self.level2 = nn.Sequential(
            ContextGuidedBlock(64, 64, dilation_rate=2, reduction=16),
            ContextGuidedBlock(64, 64, dilation_rate=2, reduction=16),
            ContextGuidedBlock(64, 64, dilation_rate=2, reduction=16),
            ContextGuidedBlock(64, 64, dilation_rate=2, reduction=16)
        )

        self.level3_0 = ContextGuidedBlock(64, 128, dilation_rate=4, reduction=16)
        self.level3 = nn.Sequential(
            ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16),
            ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16),
            ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16),
            ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16)
        )

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.level1(x)
        x = self.level2_0(x)
        x = self.level2(x)
        x = self.level3_0(x)
        x = self.level3(x)
        x = self.classifier(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x

class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.padding = Wrap(padding=padding)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, groups=nIn, bias=False, dilation=d)

    def forward(self, input):
        output = self.padding(input)
        output = self.conv(output)
        return output
    