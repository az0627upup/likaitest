import time
import multiprocessing
import os
def drink(num, name):
    print("喝汤的进程ID:", os.getpid())
    print("喝汤的主进程ID:", os.getppid())
    for i in range(num):
        print(name + "喝一口……")
        time.sleep(1)


def eat(num, name):
    print("吃饭的进程ID:", os.getpid())
    print("吃饭的主进程ID:", os.getppid())
    for i in range(num):
        print(name + "吃一口……")
        time.sleep(1)


if __name__ == '__main__':
    eat_process = multiprocessing.Process(target=eat, args=(3, "giao"))
    drink_process = multiprocessing.Process(target=drink, kwargs={"num": 4, "name": "giao"})
    print("主进程ID:", os.getpid())
    eat_process.start()
    drink_process.start()


