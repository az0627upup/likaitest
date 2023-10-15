import pandas as pd
path = './hello.txt'   #  ../代表上一级目录，  ./代表当前目录
fs = open(path, encoding='utf-8')
print(fs.readlines())
fs.close()
file_path = '../datasets/glass/glass.csv'
glass = pd.read_csv(file_path)
print(glass)
# path2 = 'E:/pythonProject/Lco/Test/hello.txt'  # 绝对路径
path2 = r'E:\pythonProject\Lco\Test\hello.txt'  # 其中r是禁止字符串转义
fs = open(path2, "r+", encoding='utf-8')
print(fs.read())
