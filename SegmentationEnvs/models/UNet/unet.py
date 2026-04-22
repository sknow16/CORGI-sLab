# from monai.networks.nets import UNet

# model = UNet(
#             spatial_dims=2,
#             in_channels=1,          # RGB画像なら3、グレーなら1
#             out_channels=3,          # 背景 + 前景
#             channels=(32, 64, 128, 256),
#             strides=(2, 2, 2),        # len = len(channels)-1
#             num_res_units=2,          # 0でシンプルなconv×2
#             norm="batch",             # batch / instance / group
#             act="relu",               # relu / leakyrelu / prelu
#             dropout=0.1,
# )