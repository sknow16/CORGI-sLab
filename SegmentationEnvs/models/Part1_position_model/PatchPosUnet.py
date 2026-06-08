from models.Part1_position_model import UNet_128_2D, UNet_64_2D, UNet_32_2D
from torch import nn
class PatchPosUnet(nn.Module):
    def __init__(self, crop_size, model_kwargs):
        super().__init__()
        if crop_size == 128:
            self.model = UNet_128_2D(**model_kwargs)
        elif crop_size == 64:
            self.model = UNet_64_2D(**model_kwargs)
        elif crop_size == 32:
            self.model = UNet_32_2D(**model_kwargs)
        else:
            raise ValueError("Invalid crop size. Supported sizes are: 128, 64, 32.")
    def forward(self, x, timesteps, cond):
        return self.model(x, timesteps, cond)