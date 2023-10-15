from time import time, sleep

def fun_one():
    start = time()
    sleep(1)
    end = time()
    cost_time = end - start
    print("func one run time {}".format(cost_time))


def fun_two():
    start = time()
    sleep(1)
    end = time()
    cost_time = end - start
    print("func two run time {}".format(cost_time))


def fun_three():
    start = time()
    sleep(1)
    end = time()
    cost_time = end - start
    print("func three run time {}".format(cost_time))
# fun_one()


def run_time(func):
    def wrapper():
        start = time()
        func()  # 函数在这里运行，执行的是@run_time下面的函数def fun_one():
        end = time()
        cost_time = end - start
        print("func run time {}".format(cost_time))
    return wrapper
@run_time
def fun_one():
    sleep(3)
    print("函数在这里运行！")
@run_time
def fun_two():
    sleep(1)
    print("函数在这里运行！")
@run_time
def fun_three():
    sleep(1)
    print("函数在这里运行！")
fun_one()
print("hello")








