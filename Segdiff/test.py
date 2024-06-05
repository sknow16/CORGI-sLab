import pandas as pd

path = "/root/save/dataset/ISBI2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv"
df = pd.read_csv(path,encoding='gbk')
print(df)

name_df = df.iloc[:, 0].tolist()
print(name_df[9])
