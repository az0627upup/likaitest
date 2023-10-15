path = './hello.txt'   # ./表示当前目录
fs = open(path, "r+", encoding='utf-8')
print(fs.read())
