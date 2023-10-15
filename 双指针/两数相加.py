class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def addTwoNumbers(l1, l2):
    # 初始化链表
    head = tree = ListNode()
    val = tmp = 0
    # 当三者有一个不为空时继续循环
    while tmp or l1 or l2:
        val = tmp
        if l1:
            val = l1.val + val
            l1 = l1.next
        if l2:
            val = l2.val + val
            l2 = l2.next

        tmp = val // 10
        val = val % 10

        # 实现链表的连接
        tree.next = ListNode(val)
        tree = tree.next

    return head.next


if __name__ == '__main__':
    lis1 = [3, 2, 4]
    lis2 = [4, 6, 5]
    L1 = headlis1 = ListNode()
    L2 = headlis2 = ListNode()
    for i in lis1:
        L1.next = ListNode(i)
        L1 = L1.next
    headlis1 = headlis1.next
    for j in lis2:
        L2.next = ListNode(j)
        L2 = L2.next
    headlis2 = headlis2.next
    m = addTwoNumbers(headlis1, headlis2)
    print(m.val)
    print(m.next.val)
    print(m.next.next.val)










