f = open("./example_3.txt", "r", encoding='utf-8')
line = f.readline()
while(line):
    print(line)
    line = f.readline()
f.close()