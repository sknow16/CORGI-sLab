import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# 画像が入ってるデータベース
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import japanize_matplotlib

# ダウンロード先ディレクトリ名
data_root = './data'

transform = transforms.Compose([
    # データのテンソル化
    transforms.ToTensor(),
    # データの正規化
    transforms.Normalize(0.5, 0.5),
    # テンソルの1階テンソル化
    transforms.Lambda(lambda x: x.view(-1)),
])

train_set = datasets.CIFAR10(
    root = data_root,
    train = True,
    download = True,
    transform = transform)

test_set = datasets.CIFAR10(
    root = data_root,
    train = False,
    download = True,
    transform = transform)

# ミニバッチサイズの指定
batch_size = 64

# 訓練データは過学習を回避するためにシャッフルする
# テストデータは対照実験のためシャッフルしない
train_loader = DataLoader(
    train_set,batch_size = batch_size,shuffle=True)

# 検証用データローダー
test_loader = DataLoader(
    test_set, batch_size = batch_size,shuffle = False)

# データローダーからバッチを取得
for images, labels in train_loader:
    break

image, label = train_set[0]
# 入力次元数
n_input = image.shape[0]

# 出力次元数
# 分類先クラス数　今回は10になる
n_output = len(set(list(labels.data.numpy())))

# 隠れ層のノード
n_hidden = 128
# 入力10出力1隠れ層のニューラルネットワークモデル
# モデルの定義
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        # 隠れ層の定義（隠れ層のノード数：n_hidden）
        self.l1 = nn.Linear(n_input, n_hidden)
        self.l12 = nn.Linear(n_hidden, n_hidden)
        # 出力層の定義
        self.l2 = nn.Linear(n_hidden, n_output)
        # ReLU関数の定義
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l12(x2)
        x4 = self.relu(x3)
        x5 = self.l2(x4)
        return x5

# 乱数の固定化
torch.manual_seed(123)
torch.cuda.manual_seed(123)

# モデルの変数生成
net = Net(n_input, n_output, n_hidden)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# モデルをGPUに送る
net = net.to(device)

# 学習率
lr = 0.01
# モデル初期化
net = Net(n_input, n_output, n_hidden).to(device)
# 損失関数:交差エントロピー
criterion = nn.CrossEntropyLoss()
# 最適化関数：勾配降下法
optimaizer = optim.SGD(net.parameters(), lr=lr)

# 繰返し回数
num_epochs = 30

# 評価結果記録用(修正)
history = np.zeros((num_epochs,5))

for epoch in range(num_epochs):
    train_acc, train_loss = 0,0
    val_acc, val_loss = 0,0
    n_train, n_test = 0,0

    # 訓練
    for inputs, labels in tqdm(train_loader):
        n_train += len(labels)
        # GPUへ転送
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 勾配の初期化
        optimaizer.zero_grad()

        # 予測計算
        outputs = net(inputs)

        # 予測ラベルの算出 (追加)
        # 予測結果の値が最も多いラベルを取り出す。これはどれだけNNの精度が良いか見るために使う。
        predicted_train = torch.max(outputs,1)[1].long()

        # 損失関数
        loss = criterion(outputs, labels)

        #勾配計算
        loss.backward()

        # パラメータ修正
        optimaizer.step()
        # # 損失と精度の計算 (追加)
        train_loss += loss.item()
        train_acc  += (predicted_train == labels).sum().item()

    # 予測フェーズ
    for inputs_test, labels_test in test_loader:
        n_test += len(labels_test)

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)

        # 予測計算
        outputs_test = net(inputs_test)

        # 損失計算
        loss_test = criterion(outputs_test, labels_test)

        # 予測データ導出
        predicted_test = torch.max(outputs_test, 1)[1]

        # 損失と精度の計算
        val_loss += loss_test.item()
        val_acc += (predicted_test == labels_test).sum().item()
    # 1エポックごとの平均誤差、平均精度 (追加)
    train_loss /= len(train_loader)
    train_acc  /= n_train
    val_loss   /= len(test_loader)
    val_acc    /= n_test
    print(f"train_loss:{train_loss}")
    print(f"train_acc :{train_acc}")
    print(f"val_loss:{val_loss}")
    print(f"val_acc :{val_acc}")
    history[epoch]=[epoch,train_loss, train_acc, val_loss, val_acc]

# 学習曲線の表示（損失）
plt.plot(history[:,0], history[:,1], 'b', label='訓練')
plt.plot(history[:,0], history[:,3], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('損失')
plt.title('学習曲線（損失）')
plt.legend()
plt.savefig('損失2.png')
plt.show()

# 学習曲線の表示（精度）
plt.plot(history[:,0], history[:,2], 'b', label='訓練')
plt.plot(history[:,0], history[:,4], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('精度')
plt.title('学習曲線（精度）')
plt.legend()
plt.savefig('精度2.png')
plt.show()

# 損失と精度の確認
print(f'初期状態：損失：{history[0,3]:.5f} 精度：{history[0,4]:.5f}')
print(f'最終状態：損失：{history[-1,3]:.5f} 精度：{history[-1,4]:.5f}')