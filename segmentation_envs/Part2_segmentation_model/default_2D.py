import torch.nn as nn
import torch
import math

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

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim,dropout=0, useconv=False):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        
        if in_ch == out_ch:
            self.skip_connection = nn.Identity()
        elif useconv:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(in_ch, out_ch, kernel_size=1)
            
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(out_ch, out_ch, 3, padding=1)
            ),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_ch),
        )

        self.time_emb = nn.Linear(time_embed_dim, 2*out_ch)

    def forward(self, x, v):
        h = self.in_layer(x)
        if v is not None:
            emb_out = self.time_emb(v)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            
            out_norm, out_rest = self.out_layer[0], self.out_layer[1:]
            shift, scale = emb_out.chunk(2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = self.out_layer(h)
        return h+self.skip_connection(x)
    
# U-Net構造 ====================================================================
# time_embed_dim=100：各タイムステップに対応する特徴量の値(実際の時間とは関係ない)
# サイズ縮小：Maxプーリング　サイズ拡大：バイリニア補間
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, cond_in_ch=3 ,hidden_ch=32, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.position_encoder = PositionEncoder(cond_in_ch, hidden_ch)
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3,1,1)  # パッチ画像(1, 64, 64) -> (32, 64, 64)に変更
        self.down1 = ResBlock(hidden_ch+hidden_ch*4, hidden_ch*2, time_embed_dim)
        self.down2 = ResBlock(hidden_ch*2+hidden_ch*8, hidden_ch*4, time_embed_dim)
        self.bot1  = ResBlock(hidden_ch*4+hidden_ch*8, hidden_ch*8, time_embed_dim)
        self.up2 = ResBlock(hidden_ch*4 + hidden_ch*8, hidden_ch*4, time_embed_dim)
        self.up1 = ResBlock(hidden_ch*4 + hidden_ch*2, hidden_ch*2, time_embed_dim)
        self.out = nn.Conv2d(hidden_ch*2, out_ch, 1)
        
        self.maxpool  = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps, cond):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device) # 拡散ステップのためのposition encoder
        cond1, cond2, cond3 = self.position_encoder(cond)
        x = self.conv_in(x)
        x = torch.concat([x, cond1], dim=1)
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x = torch.concat([x, cond2], dim=1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)
        x = torch.concat([x, cond3], dim=1)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x
    
# x = torch.randn(1,3,64,64)
# t = torch.tensor([1])
# cond = torch.randn(1,3,256,256)
# model = UNet()
# output = model(x,t,cond)
# print(output.shape)