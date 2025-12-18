import torch
from torch import nn
import torch
from torch import nn
import torch.nn.functional as F
from kan.ukan_3d_utils import ConvLayer3D, D_ConvLayer3D, PatchEmbed3D, KANBlock3D

class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=(64,256,256), patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.encoder1 = ConvLayer3D(input_channels, kan_input_dim//8)  
        self.encoder2 = ConvLayer3D(kan_input_dim//8, kan_input_dim//4)  
        self.encoder3 = ConvLayer3D(kan_input_dim//4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock3D(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.block2 = nn.ModuleList([KANBlock3D(
            dim=embed_dims[2],
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])

        self.dblock1 = nn.ModuleList([KANBlock3D(
            dim=embed_dims[1], 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer
            )])

        self.dblock2 = nn.ModuleList([KANBlock3D(
            dim=embed_dims[0], 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer
            )])
        # self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed3D(img_size=(img_size[0]//4, img_size[1]//4, img_size[2]//4), patch_size=(3,3,3), stride=(2,2,2), in_chans=embed_dims[0], embed_dim=embed_dims[1])
        # self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed3D(img_size=(img_size[0]//8, img_size[1]//8, img_size[2]//8), patch_size=(3,3,3), stride=(2,2,2), in_chans=embed_dims[1], embed_dim=embed_dims[2])
        
        self.decoder1 = D_ConvLayer3D(embed_dims[2], embed_dims[1])  
        self.decoder2 = D_ConvLayer3D(embed_dims[1], embed_dims[0])  
        self.decoder3 = D_ConvLayer3D(embed_dims[0], embed_dims[0]//4) 
        self.decoder4 = D_ConvLayer3D(embed_dims[0]//4, embed_dims[0]//8)
        self.decoder5 = D_ConvLayer3D(embed_dims[0]//8, embed_dims[0]//8)
        self.final = nn.Conv3d(embed_dims[0]//8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        
        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        
        ### Stage 2
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out

        ### Stage 3
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out

        ### Tokenized KAN Stage
        ### Stage 4
        out, D, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, D, H, W)
        out = self.norm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        ### Bottleneck
        out, D, H, W= self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, D, H, W)
        out = self.norm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        ### Stage 4
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2,2,2),mode ='trilinear'))

        out = torch.add(out, t4)
        _, _, D, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, D, H, W)

        ### Stage 3
        out = self.dnorm3(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        _,_, D, H, W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, D, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))

        return self.final(out)
    
# x = torch.randn(1,3,64,256,256)
# model = UKAN(num_classes=1, img_size=(64, 256, 256), patch_size=16, input_channels=3, embed_dims=[256, 320, 512], depths=[1, 1], no_kan=False)
# model.eval()
# with torch.no_grad():
#     out = model(x)
# print(out.shape)  # Expected output shape: (2, 10, 224, 224)
