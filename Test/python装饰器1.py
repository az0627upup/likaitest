import logging
def log(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}' with arguments {args} {kwargs}")
        result = func(*args, **kwargs)
        print(f"Function '{func.__name__}' returned {result}")
        return result
    return wrapper

@log
def my_function(x, y):
    return x + y
my_function(3, 5)

# 在上面的代码中，log 是一个装饰器函数，它接受一个函数作为参数并返回一个新的函数 wrapper。
# wrapper 函数在执行原始函数之前和之后打印一条日志，并将结果返回。@log 是一个装饰器的语法糖，
# 它等价于执行 my_function = log(my_function)，将 my_function 函数传递给 log 装饰器，
# 然后将返回的 wrapper 函数重新赋值给 my_function，从而实现了在函数执行前后打印日志的功能。






# def use_logging(func):
#     logging.warning("%s is running")
# def foo():
#     print('i am foo')
# use_logging(foo())
