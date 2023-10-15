from collections import defaultdict
f = open("a.txt", "r")
line = f.readline()
dict1 = defaultdict()
while line:
    if line not in dict1.keys():
        dict1[line] = 0
    else:
        dict1[line] += 1
    line = f.readline()
list1 = []
for key,value in dict1.items():
    list1.append((key, value))
result = sorted(list1, key=lambda x:x[1], reverse=True)
print(result[:10])
f.close()