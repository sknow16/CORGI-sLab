import torch.nn as nn
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder

class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 4, 2, [64, 64, 128, 256, 512, 64])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])

        self.model = BasicUNetDe(3, 3+4, 3,  [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

    def forward(self, x=None, t=None,image=None):
        embeddings = self.embed_model(image)
        return self.model(x, t=t, image=image, embeddings=embeddings)
    
import torch
x = torch.randn(1,3,96,96,96)
image=torch.randn(1,4,96,96,96)
step = torch.tensor([0])
model = DiffUNet()
model_kwargs=dict(image=image)
y = model(x=x,t=step, **model_kwargs)

print(y.shape)