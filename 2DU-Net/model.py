import torch
import torch.nn as nn
import torch.nn.functional as F

class encoder(nn.Module):
    def __init__(self, in_channel, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

        self.conv3 = nn.Conv2d(ch, ch*2, 3, 1, 1)
        self.conv4 = nn.Conv2d(ch*2, ch*2, 3, 1, 1)
        self.conv5 = nn.Conv2d(ch*2, ch*4, 3, 1, 1)
        self.conv6 = nn.Conv2d(ch*4, ch*4, 3, 1, 1)

        self.maxpool = nn.MaxPool2d(2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x1 = self.relu(self.conv2(x))
        x = self.maxpool(x1)
        x = self.relu(self.conv3(x))
        x2 = self.relu(self.conv4(x))
        x = self.maxpool(x2)
        x = self.relu(self.conv5(x))
        x3 = self.relu(self.conv6(x))
        x = self.maxpool(x3)
        return x, x1, x2, x3
    
class bottle_neck(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch*4, ch*8, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch*8, ch*8, 3, 1, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x


class decoder(nn.Module):
    def __init__(self, out_channel, ch):
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(ch*8, ch*4, 2, 2)
        self.upconv2 = nn.ConvTranspose2d(ch*4, ch*2, 2, 2)
        self.upconv3 = nn.ConvTranspose2d(ch*2, ch, 2, 2)

        self.conv1 = nn.Conv2d(ch*8, ch*4, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch*4, ch*4, 3, 1, 1)
        self.conv3 = nn.Conv2d(ch*4, ch*2, 3, 1, 1)
        self.conv4 = nn.Conv2d(ch*2, ch*2, 3, 1, 1)
        self.conv5 = nn.Conv2d(ch*2, ch, 3, 1, 1)
        self.conv6 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv7 = nn.Conv2d(ch, out_channel, 3, 1, 1)

        self.relu = nn.ReLU()

    def forward(self, x, x1, x2, x3):
        x = self.upconv1(x)
        x = torch.cat([x, x3], dim = 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.upconv2(x)
        x = torch.cat([x, x2], dim = 1)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.upconv3(x)
        x = torch.cat([x, x1], dim = 1)
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.conv7(x)
        return x



class unet(nn.Module):
    def __init__(self, in_channel, out_channel, ch):
        super().__init__()
        self.encoder = encoder(in_channel, ch)
        self.bottle_neck = bottle_neck(ch)
        self.decoder = decoder(out_channel, ch)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x, x1, x2, x3)

        return x

model = unet(1, 1, 64)
model.eval()

x = torch.randn(1, 1, 256, 256)
with torch.no_grad():
  y = model(x)
  print(y.shape)

