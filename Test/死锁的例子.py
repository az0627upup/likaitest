import threading
import time
# 创建两个锁
lock1 = threading.Lock()
lock2 = threading.Lock()

# 定义线程1，尝试获取锁1后再获取锁2
def thread1():
    print("Thread 1: Attempting to acquire lock 1")
    lock1.acquire()
    print("Thread 1: Acquired lock 1, attempting to acquire lock 2")
    time.sleep(1)   # 一定要设置时间等待，不然不会死锁
    lock2.acquire()
    print("Thread 1: Acquired lock 2")
    # 执行一些工作
    lock2.release()
    lock1.release()

# 定义线程2，尝试获取锁2后再获取锁1
def thread2():
    print("Thread 2: Attempting to acquire lock 2")
    lock2.acquire()
    print("Thread 2: Acquired lock 2, attempting to acquire lock 1")
    time.sleep(1)
    lock1.acquire()
    print("Thread 2: Acquired lock 1")
    # 执行一些工作
    lock1.release()
    lock2.release()

# 创建两个线程并启动它们
t1 = threading.Thread(target=thread1)
t2 = threading.Thread(target=thread2)
t1.start()
t2.start()
t1.join()
t2.join()

print("Both threads have finished")
