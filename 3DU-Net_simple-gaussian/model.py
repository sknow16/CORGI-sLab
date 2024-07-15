import torch
import torch.nn as nn
import math
# ゼロつく同様のtime embedding
# Position Encoding Functions
def _pos_encoding(time_idx, output_dim, device='cpu'):
    t, D = time_idx, output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)
    div_term = torch.exp(i / D * math.log(10000))

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])
    return v

def pos_encoding(timesteps, output_dim, device='cpu'):
    batch_size = len(timesteps)
    device = timesteps.device
    v = torch.zeros(batch_size, output_dim, device=device)
    for i in range(batch_size):
        v[i] = _pos_encoding(timesteps[i], output_dim, device)
    return v

class DoubleConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,time_embed_dim=100):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 追加
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels)
        )

    def forward(self, x, timedbed):
        # 追加
        N,C,_,_,_ = x.shape
        t = self.mlp(timedbed)
        t = t.view(N, C, 1, 1, 1)
        # x を x+tに変更
        return self.double_conv(x+t)

class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels,time_embed_dim=100,cond_channels=4):
        super(UNet3D, self).__init__()
        self.time_embed_dim = time_embed_dim
        
        self.enc1 = DoubleConv(in_channels+cond_channels, 32, 64,time_embed_dim)
        self.enc2 = DoubleConv(64, 64, 128,time_embed_dim)
        self.enc3 = DoubleConv(128, 128, 256,time_embed_dim)

        self.down = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(256, 256, 512,time_embed_dim)

        self.upconv3 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256+512, 256, 256,time_embed_dim)
        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128+256, 128, 128,time_embed_dim)
        self.upconv1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64+128, 64, 64,time_embed_dim)

        self.conv_last = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x, t, y):
        t = pos_encoding(t, self.time_embed_dim, device=x.device)
        
        x = torch.cat([x,y],dim=1)
        x0 = self.enc1(x,t)
        x1 = self.enc2(self.down(x0), t)
        x2 = self.enc3(self.down(x1), t)

        x3 = self.bottleneck(self.down(x2), t)

        x4 = self.upconv3(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.dec3(x4,t)

        x5 = self.upconv2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.dec2(x5,t)

        x6 = self.upconv1(x5)
        x6 = torch.cat([x6, x0], dim=1)
        x6 = self.dec1(x6,t)

        return self.conv_last(x6)

# 動作確認
# unet = UNet3D(1,3)
# a    = torch.randn(1,1,32,32,32)
# y    = torch.randn(1,4,32,32,32)
# t   = torch.randn(1)
# b = unet(a,t,y)
# print(b.shape)