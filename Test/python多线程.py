import threading
import time

def eat(name):
    for i in range(4):
        print(name + "我吃……")
        time.sleep(0.5)


def drink(name, count):
    for i in range(count):
        print(name + "我喝……")
        time.sleep(0.5)


if __name__ == '__main__':
    eat_thread = threading.Thread(target=eat, args=("giao",), daemon=True)
    drink_thread = threading.Thread(target=drink, kwargs={"name": "qz", "count": 4}, daemon=True)
    eat_thread.start()
    drink_thread.start()
    time.sleep(1)
    print("我吃不下也喝不下了！")


