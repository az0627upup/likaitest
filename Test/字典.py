# 第一种方法
dic1 = {'name': 'hacker', 'age': '18'}
# 第二种方法
dic2 = dict(name='hacker',age='18')
# 第三种方法
dic3 = dict([('name','hacker'),('age','18')])
car = {"brand": "Porsche", "model": "911", "year": 1963}

for dic in dic1.items():
    print(dic)
print(dic1.keys())
print(dic1.values())
print(dic1.get("name"))
car.pop("model")
print(car)
