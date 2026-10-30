import torch.nn as nn
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet_denose_diffunet_ref import BasicUNetDe as BasicUNetDe_ref
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose_kan_12 import BasicUNetDe as BasicUNetDe_kan_12
from unet.basic_unet_denose_kan_23 import BasicUNetDe as BasicUNetDe_kan_23
from unet.basic_unet_denose_1shift import BasicUNetDe as BasicUNetDe_1shift
from unet.basic_unet_denose_1shift_bottle import BasicUNetDe as BasicUNetDe_1shift_bottle
class DiffUKAN(nn.Module):
    def __init__(self, img_size=(64,240,240)) -> None:
        super().__init__()
        # self.embed_model = BasicUNetEncoder(3, 4, 2, [64, 64, 128, 256, 512, 64])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])
        self.embed_model = BasicUNetEncoder(3, 4, 2, [32, 32, 64, 128, 160, 32])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])

        # self.model = BasicUNetDe(3, 3+4, 3,  [64, 64, 128, 256, 512, 64], 
        self.model = BasicUNetDe(3, 3+4, 3,  [32, 32, 64, 128, 160, 32], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),img_size=img_size)

    def forward(self, x=None, t=None,image=None):
        embeddings = self.embed_model(image)
        return self.model(x, t=t, image=image, embeddings=embeddings)

class DiffUKAN_1shift(nn.Module):
    def __init__(self, img_size=(64,240,240)) -> None:
        super().__init__()
        # self.embed_model = BasicUNetEncoder(3, 4, 2, [64, 64, 128, 256, 512, 64])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])
        self.embed_model = BasicUNetEncoder(3, 4, 2, [32, 32, 64, 128, 160, 32])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])

        # self.model = BasicUNetDe(3, 3+4, 3,  [64, 64, 128, 256, 512, 64], 
        self.model = BasicUNetDe_1shift(3, 3+4, 3,  [32, 32, 64, 128, 160, 32], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),img_size=img_size)

    def forward(self, x=None, t=None,image=None):
        embeddings = self.embed_model(image)
        return self.model(x, t=t, image=image, embeddings=embeddings)
    
# 1shiftしてボトルネックにもKANを入れたモデル
class DiffUKAN_1shift_bottle(nn.Module):
    def __init__(self, img_size=(64,240,240)) -> None:
        super().__init__()
        # self.embed_model = BasicUNetEncoder(3, 4, 2, [64, 64, 128, 256, 512, 64])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])
        self.embed_model = BasicUNetEncoder(3, 4, 2, [32, 32, 64, 128, 160, 32])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])

        # self.model = BasicUNetDe(3, 3+4, 3,  [64, 64, 128, 256, 512, 64], 
        self.model = BasicUNetDe_1shift_bottle(3, 3+4, 3,  [32, 32, 64, 128, 160, 32], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),img_size=img_size)

    def forward(self, x=None, t=None,image=None):
        embeddings = self.embed_model(image)
        return self.model(x, t=t, image=image, embeddings=embeddings)
    
# DiffUKANのKANブロックをエンコーダ最下層とボトルネック部のみ
class DiffUKAN_12(nn.Module):
    def __init__(self, img_size=(64,240,240)) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 4, 2, [32, 32, 64, 128, 160, 32])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])
        self.model = BasicUNetDe_kan_12(3, 3+4, 3,  [32, 32, 64, 128, 160, 32], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),img_size=img_size)

    def forward(self, x=None, t=None,image=None):
        embeddings = self.embed_model(image)
        return self.model(x, t=t, image=image, embeddings=embeddings)

class DiffUKAN_23(nn.Module):
    def __init__(self, img_size=(64,240,240)) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(3, 4, 2, [32, 32, 64, 128, 160, 32])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])
        self.model = BasicUNetDe_kan_23(3, 3+4, 3,  [32, 32, 64, 128, 160, 32], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False}),img_size=img_size)

    def forward(self, x=None, t=None,image=None):
        embeddings = self.embed_model(image)
        return self.model(x, t=t, image=image, embeddings=embeddings)
    
# DiffUKAN_wo_KANはKANを使わないモデル
class DiffUKAN_wo_KAN(nn.Module):
    def __init__(self, img_size=(64,240,240)) -> None:
        super().__init__()
        # self.embed_model = BasicUNetEncoder(3, 4, 2, [64, 64, 128, 256, 512, 64])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])
        self.embed_model = BasicUNetEncoder(3, 4, 2, [32, 32, 64, 128, 160, 32])  # (次元数, 入力のチャネル数, 出力チャネル数(使ってない), [モデルの階層の出力チャネル数(64は使ってない)])

        # self.model = BasicUNetDe(3, 3+4, 3,  [64, 64, 128, 256, 512, 64], 
        self.model = BasicUNetDe_ref(3, 3+4, 3,  [32, 32, 64, 128, 160, 32], 
                                act = ("LeakyReLU", {"negative_slope": 0.1, "inplace": False})) #,img_size=img_size)

    def forward(self, x=None, t=None,image=None):
        embeddings = self.embed_model(image)
        return self.model(x, t=t, image=image, embeddings=embeddings)
    
# import torch
# x = torch.randn(1,3,64,240,240)
# image=torch.randn(1,4,64,240,240)
# step = torch.tensor([0])
# model = DiffUKAN_1shift_bottle(img_size=(64,240,240))
# model_kwargs=dict(image=image)
# y = model(x=x,t=step, **model_kwargs)
# print("paramters:", sum(p.numel() for p in model.parameters()))
# # Milion表記
# print(f"モデルのパラメータ数: {sum(p.numel() for p in model.parameters())/1e6:.2f} Million")
# print(y.shape)