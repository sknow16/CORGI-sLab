import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.enc1 = DoubleConv(in_channels, 32, 64)
        self.enc2 = DoubleConv(64, 64, 128)
        self.enc3 = DoubleConv(128, 128, 256)

        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(256, 256, 512)

        self.upconv3 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256+512, 256, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128+256, 128, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64+128, 64, 64)

        self.conv_last = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.enc1(x)
        x1 = self.enc2(self.down(x0))
        x2 = self.enc3(self.down(x1))

        x3 = self.bottleneck(self.down(x2))

        x4 = self.upconv3(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.dec3(x4)

        x5 = self.upconv2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.dec2(x5)

        x6 = self.upconv1(x5)
        x6 = torch.cat([x6, x0], dim=1)
        x6 = self.dec1(x6)

        return self.conv_last(x6)


