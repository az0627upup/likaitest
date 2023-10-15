class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def createlistnode(lis):
    head = ListNode(lis[0])
    p = head
    for i in range(1, len(lis)):
        tem_lis = ListNode(lis[i])
        p.next = tem_lis
        p = p.next
    return head
def printlistnode(head):
    while(head):
        print(head.val, end=' ')
        head = head.next


class Solution:
    def sortList(self, head):
        def sortfunc(head, tail):
            if not head:
                return head
            if head.next == tail:
                head.next = None
                return head
            slow = fast = head
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next
            mid = slow
            return merge(sortfunc(head, mid), sortfunc(mid, tail))

        def merge(head1, head2):
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            return dummyHead.next
        return sortfunc(head, None)

if __name__ =='__main__':
    lis = [7, 9, 77, 56, 1, 23]
    head = createlistnode(lis)
    solution = Solution()
    head = solution.sortList(head)
    printlistnode(head)
