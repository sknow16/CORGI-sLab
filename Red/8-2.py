import torch

# GPUに繋がっているかの確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)