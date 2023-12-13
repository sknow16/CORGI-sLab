import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ReLU関数のグラフ
relu = nn.ReLU()
x_np = np.arange(-2, 2.1, 0.25)
x = torch.tensor(x_np).float()
y = relu(x)

plt.plot(x.data, y.data)
plt.title('ReLU')
plt.savefig('ReLU.png')
plt.show()