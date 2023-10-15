class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def create_linkList_tail(lis):  # 尾插法
    head = ListNode(lis[0])
    tail = head  # 头结点
    for i in lis[1:]:
        new_node = ListNode(i)  # 让新的数据结构化，然后赋给新结点
        tail.next = new_node  # 让新结点赋给尾结点的下一个结点
        tail = new_node  # 刚来的新结点成为新的尾结点
    return head


def print_linkList(link):  # 始终要记得link是一个结构，调用时从头结点开始
    while link:  # 如果结点存在
        print(link.val, end=' ')  # 输出当前结点的数据域
        link = link.next  # 让下一个结点成为下一次要操作的结点


def removeNthFromEnd(head, n):
    def getLength(head):     # 求出链表的长度
        length = 0
        while head:
            length += 1
            head = head.next
        return length
    dummy = ListNode(0, head)
    length = getLength(head)
    cur = dummy
    for i in range(1, length - n + 1):
        cur = cur.next
    cur.next = cur.next.next
    return dummy.next

if __name__ == '__main__':
    lis = [1, 10, 9, 8, 7, 5, 3, 22, 7, 45, 3]
    head = create_linkList_tail(lis)
    ne = removeNthFromEnd(head, 3)
    print_linkList(ne)



