import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# 乱数固定
def torch_seed(seed=123):   
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
torch_seed()
data_root = "./data" # ダウンロード先のディレクト名

# データの前処理
transform = transforms.Compose([
    transforms.ToTensor(), # データをテンソル化処理
    transforms.Normalize(0.5,0.5), # データの正規化 範囲を[-1,1]に変更できる
    transforms.Lambda(lambda x: x.view(-1)) # 1次元に変更するlamda関数を導入することでtransformsにない処理を可能にする.
])

train_set = datasets.MNIST(
    root  = data_root,
    train = True,     # 訓練データかテストデータか
    download  = True, # 元データがない場合ダウンロードする
    transform = transform
)
test_set  = datasets.MNIST(
    root  = data_root,
    train = False,
    download  = True,
    transform = transform
)

batch_size = 64 # ミニバッチのサイズ指定
# 訓練用データローダ―
train_loader = DataLoader(
    train_set,
    batch_size = batch_size,
    shuffle    = True # データにシャフルを加える
)
# テスト用データローダ―
test_loader = DataLoader(
    test_set,
    batch_size = batch_size,
    shuffle    = False
)

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super().__init__()
        self.L1 = nn.Linear(input_size,hidden_size)
        self.L2 = nn.Linear(hidden_size,output_size)
        self.activation = nn.ReLU(inplace=True) # inplace = 入力を保存せず出力に置き換える
    def forward(self,x):
        x1 = self.L1(x)
        x2 = self.activation(x1)
        x3 = self.L2(x2)
        return x3

input  = 784
hidden = 128
output = 10

lr = 0.001
epochs = 10
history = np.zeros((epochs,5)) # 記録用配列 (epoch,train_loss,train_acc,val_loss,val_acc)

net = Net(input,hidden,output)
# criterion = nn.BCELoss()                      # 損失関数: 2値交差エントロピー
criterion = nn.CrossEntropyLoss()               # 損失関数: 交差エントロピーとsoftmax関数が組み合わさっている (ミニバッチ内の平均損失が算出される)
optimizer = optim.SGD(net.parameters(),lr = lr) # 最適化関数: 勾配降下法(SGD)

for epoch in range(epochs):
    print(f'{epoch+1}エポック')
    train_acc,train_loss = 0, 0
    val_acc, val_loss    = 0, 0
    n_train, n_test      = 0, 0
    # 学習
    for inputs, labels in train_loader:
        n_train += len(labels)
        optimizer.zero_grad() # 勾配の初期化
        outputs = net(inputs) # 予測計算
        predicted_train = torch.max(outputs,1)[1].long()   # 予測ラベルの算出
        
        loss            = criterion(outputs,labels) # 損失関数
        loss.backward()  # 勾配計算
        optimizer.step() # パラメータの更新
        train_loss += loss.item()
        train_acc  += (predicted_train == labels).sum().item()
        
        
    for inputs, labels in test_loader:
        n_test += len(labels)
        outputs_test = net(inputs)
        loss_test    = criterion(outputs_test,labels)
        predicted_test = torch.max(outputs_test,1)[1].long()
        # 損失と精度の計算
        val_loss += loss_test.item()
        val_acc  += (predicted_test == labels).sum().item()

    # 1エポックごとの平均誤差、平均精度
    train_loss /= len(train_loader)
    train_acc  /= n_train
    val_loss   /= len(test_loader)
    val_acc    /= n_test
    print(f"train_loss:{train_loss}")
    print(f"train_acc :{train_acc}")
    print(f"val_loss:{val_loss}")
    print(f"val_acc :{val_acc}")
    history[epoch] = [epoch + 1, train_loss, train_acc, val_loss, val_acc]
print(f"初期状態:損失:{history[0,3]:.5f} 精度:{history[0,4]:5f}")
print(f"最終状態:損失:{history[-1,3]:.5f} 精度:{history[-1,4]:5f}")

plt.plot(history[:,0],history[:,1],c='b',label="train")
plt.plot(history[:,0],history[:,3],c="k",label="test")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("学習曲線(損失)")
plt.legend()
plt.show()