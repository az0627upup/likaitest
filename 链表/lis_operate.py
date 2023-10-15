class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def create_linkList_tail(lis):  # 尾插法
    head = ListNode(lis[0])
    tail = head  # 尾结点
    for i in lis[1:]:
        new_node = ListNode(i)  # 让新的数据结构化，然后赋给新结点
        tail.next = new_node  # 让新结点赋给尾结点的下一个结点
        tail = new_node  # 刚来的新结点成为新的尾结点
    return head


def print_linkList(link):  # 始终要记得link是一个结构，调用时从头结点开始
    while link:  # 如果结点存在
        print(link.val, end=' ')  # 输出当前结点的数据域
        link = link.next  # 让下一个结点成为下一次要操作的结点


# if __name__ == '__main__':
#     m = [1, 3, 5, 6, 4, 9, 10]
#     n = list(map(int, input().split(" ")))
#     head = create_linkList_tail(n)
#     print_linkList(head)

