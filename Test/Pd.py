import numpy as np
import pandas as pd
path = "E:/pythonProject/Lco/datasets/iris/iris.csv"
# df = pd.read_csv(path, header=1,index_col=0)
df = pd.read_csv(path)
data = df.sample(frac=1).reset_index(drop=True)
fc = df.iloc[0:4]
dat = np.array(fc)
label = dat[:, -1]
print(label.tolist())

a = np.array([[4, 23, 6, 10], [9, 6, 3, 7]])
b = np.array([[1, 2, 3, 4], [10, 8, 7, 6]])














