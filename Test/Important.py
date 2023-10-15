# def inputxx():
#     lis1 = []
#     N1, M1, N2, M2 = map(int, input().split(" "))
#     for i in range(N1):
#         x = list(map(int, input().split(" ")))
#         lis1.append(x)
def inpux4():
    lis1 = []
    N1, M1 = map(int, input().split(" "))
    for i in range(N1):
        x = list(map(int, input(",")))    # x = list(map(int, input().split(",")))   #每行输入的元素以,分隔符间隔而不是空格
        lis1.append(x)


def inputx2():
    li1 = []
    for i in range(3):
        li1.append(input())
    return li1
def inputx3():
    arr = input()
    num = [int(i) for i in arr.split()]
    return arr
if __name__ == '__main__':
    # lis2 = inputx2()
    lis2 = inputx3()
    for i in lis2:
        print(i, end=" ")


