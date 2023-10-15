y = (lambda x: x**2)(2)    #就是lambda后面跟的是参数，然后是函数表达式，()里面是传递的参数的值
z = list(map((lambda x: x**2), (range(10))))
print(z)
print(y)
