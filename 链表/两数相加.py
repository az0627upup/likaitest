from lis_operate import *


def addTwoNumbers(l1,l2):
    # 初始化链表
    head = tree = ListNode()
    # val 用于记录当前的计算结果
    # temp 用于记录是否需要进位
    val = tmp = 0
    # 有进位的情况，或者 l1 不为空，或者 l2 不为空时继续循环
    # 可以理解为，最高位是通过进位产生的，需要增加一个节点
    # 并且， l1 或者 l2 不为空时，肯定需要增加一个节点
    while tmp or l1 or l2:
        # 处理上一次循环中是否有进位的问题
        val = tmp

        # 如果 l1 不为空，与 treeL.val 相加，更新相加的结果
        # 并且使l1指向下一个值
        # l2 与 l1 同理
        if l1:
            val = l1.val + val
            l1 = l1.next
        if l2:
            val = l2.val + val
            l2 = l2.next

        # tmp 取 val 数值的十位，即进位
        tmp = val // 10
        # val 保留 val 数值的个位，即本次最终的 val值
        val = val % 10

        # 将 val 的值赋给链表的下一个节点的 val 属性
        tree.next = ListNode(val)
        # 将指针移至下一个节点
        tree = tree.next

    # val 值的记录是从根节点的下一个节点开始记录的
    return head.next


if __name__ == '__main__':
    lis1 = [9,9,9,9,9,9,9]
    lis2 = [9,9,9,9]
    l1 = create_linkList_tail(lis1)
    print(lis1)
    l2 = create_linkList_tail(lis2)
    print(lis2)
    head = addTwoNumbers(l1, l2)
    # print_linkList(head)