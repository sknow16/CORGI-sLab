import torch.nn as nn
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder

class DiffUKAN(nn.Module):
    def __init__(self, img_size=(64,240,240)) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 4, 2, [64, 64, 128, 256, 512, 64])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])

        self.model = BasicUNetDe(3, 3+4, 3,  [64, 64, 128, 256, 512, 64], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),img_size=img_size)

    def forward(self, x=None, t=None,image=None):
        embeddings = self.embed_model(image)
        return self.model(x, t=t, image=image, embeddings=embeddings)
    
# import torch
# x = torch.randn(1,3,64,256,256)
# image=torch.randn(1,4,64,256,256)
# step = torch.tensor([0])
# model = DiffUKAN()
# model_kwargs=dict(image=image)
# y = model(x=x,t=step, **model_kwargs)

# print(y.shape)