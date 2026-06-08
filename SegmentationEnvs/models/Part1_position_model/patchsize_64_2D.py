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
    
# PositionEncorder構造 =========================================================
# 位置情報を条件付けするためのエンコーダー
# 256×256の画像を畳み込んで「64×64」「32×32」「16×16」「8×8」になった特徴マップを
# 学習用U-Netに条件付けする
# ==============================================================================
class PositionEncoder(nn.Module):
    # モデルの構造を定義
    def __init__(self, in_ch, hidden_ch=32):
        """
        in_ch: 入力チャンネル数
        hidden_ch: 隠れ層のチャンネル数
        """
        super().__init__()
        self.in_conv = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1)
        self.down = nn.MaxPool2d(2) # 1/2に縮小
        # nn.Sequentialで、複数の層(GroupNorm, ReLU, Conv)を順に実行するように定義
        self.conv1 = ResBlock(hidden_ch, hidden_ch*2, 100)
        # nn.Sequentialで、複数の層(GroupNorm, ReLU, Conv)を順に実行するように定義
        self.conv2 = ResBlock(hidden_ch*2, hidden_ch*4, 100)
        # # nn.Sequentialで、複数の層(GroupNorm, ReLU, Conv)を順に実行するように定義
        self.conv3 = ResBlock(hidden_ch*4, hidden_ch*4, 100)
        self.conv4 = ResBlock(hidden_ch*4, hidden_ch*8, 100)
        self.conv5 = ResBlock(hidden_ch*8, hidden_ch*8, 100)
    
    # 順伝播を定義
    def forward(self, x):
        x = self.in_conv(x) # (32, 256, 256) 
        x = self.down(x)  # (32, 256, 256) -> (32, 128, 128)
        x1 = self.conv1(x,None)  # (64, 128, 128)

        x = self.down(x1)  # (64, 128, 128) -> (64, 64, 64)
        x2 = self.conv2(x,None)  # (128, 64, 64)
        x = self.down(x2)  # (128, 64, 64) -> (128, 32, 32)
        x3 = self.conv3(x,None)  # (256, 32, 32)
        x = self.down(x3)  # (128, 32, 32) -> (128, 16, 16)
        x4 = self.conv4(x,None)  # (256, 16, 16)
        x = self.down(x4)  # (256, 16, 16) -> (256, 8, 8)
        x5 = self.conv5(x, None) # (256, 8, 8)
        return x2, x3, x4, x5

# U-Net構造 ====================================================================
# 4箇所(3層+bottleneck)にそれぞれPositionEncoderから特徴マップを
# 位置情報としてconcat(cond1,2,3,4)
# ==============================================================================
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, cond_in_ch=1 ,hidden_ch=32, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.position_encoder = PositionEncoder(cond_in_ch, hidden_ch)
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3,1,1)  # パッチ画像(1, 64, 64) -> (32, 64, 64)に変更
        self.down1 = ResBlock(hidden_ch+hidden_ch*4, hidden_ch*2, time_embed_dim)
        self.down2 = ResBlock(hidden_ch*2+hidden_ch*4, hidden_ch*4, time_embed_dim)
        self.down3 = ResBlock(hidden_ch*4+hidden_ch*8, hidden_ch*8, time_embed_dim)

        self.bot1  = ResBlock(hidden_ch*8+hidden_ch*8, hidden_ch*8, time_embed_dim)

        self.up3 = ResBlock(hidden_ch*8 + hidden_ch*8, hidden_ch*4, time_embed_dim)
        self.up2 = ResBlock(hidden_ch*4 + hidden_ch*4, hidden_ch*2, time_embed_dim)
        self.up1 = ResBlock(hidden_ch*2 + hidden_ch*2, hidden_ch, time_embed_dim)
        self.out = nn.Conv2d(hidden_ch, out_ch, 1)
        
        self.maxpool  = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, timesteps, cond):
        v = pos_encoding(timesteps, self.time_embed_dim, x.device) # 拡散ステップのためのposition encoder
        cond1, cond2, cond3, cond4 = self.position_encoder(cond)
        x = self.conv_in(x)
        x = torch.concat([x, cond1], dim=1)
        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x = torch.concat([x, cond2], dim=1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)
        x = torch.concat([x, cond3], dim=1)
        x3 = self.down3(x, v)
        x = self.maxpool(x3)
        x = torch.concat([x,cond4], dim=1)
        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x, v)

        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x
    
# x = torch.randn(1,1,64,64)
# t = torch.tensor([1])
# cond = torch.randn(1,1,256,256)
# model = UNet()
# output = model(x,t,cond)
# print(output.shape)