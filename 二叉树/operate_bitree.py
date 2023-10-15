import math


# 定义二叉树
class BTNode(object):
    """docstring for BTNode"""

    def __init__(self, data):
        self.data = data
        self.leftChild = None
        self.rightChild = None


# 插入元素
def InsertElementBinaryTree(root, node):
    if root:
        if node.data < root.data:
            if root.leftChild:
                InsertElementBinaryTree(root.leftChild, node)
            else:
                root.leftChild = node
        else:
            if root.rightChild:
                InsertElementBinaryTree(root.rightChild, node)
            else:
                root.rightChild = node
    else:
        return 0


# 初始化二叉树
def InitBinaryTree(dataSource):
    root = BTNode(dataSource[0])
    n = length(dataSource)
    for x in range(1, n):
        node = BTNode(dataSource[x])
        InsertElementBinaryTree(root, node)
    return root


print('Done...')


# 先序
def PreorderTraversalBinaryTree(root):
    # 递归调用
    if root:
        print('%d | ' % root.data, )
        PreorderTraversalBinaryTree(root.leftChild)
        PreorderTraversalBinaryTree(root.rightChild)


# 中序
def InorderTraversalBinaryTree(root):
    if root:
        InorderTraversalBinaryTree(root.leftChild)
        print('%d | ' % root.data, )
        InorderTraversalBinaryTree(root.rightChild)


# 后序
def PostorderTraversalBinaryTree(root):
    if root:
        PostorderTraversalBinaryTree(root.leftChild)
        PostorderTraversalBinaryTree(root.rightChild)
        print('%d | ' % root.data, )


# 分层
def TraversalByLayer(root, length):
    stack = []
    stack.append(root)
    for x in range(length):
        node = stack[x]
        print('%d | ' % node.data, )
        if node.leftChild:
            stack.append(node.leftChild)
        if node.rightChild:
            stack.append(node.rightChild)


if __name__ == '__main__':
    dataSource = [3, 4, 2, 6, 7, 1, 8, 5]
    length = len(dataSource)
    BTree = InitBinaryTree(dataSource)
    print('****NLR:')
    PreorderTraversalBinaryTree(BTree)
    print('\n****LNR')
    InorderTraversalBinaryTree(BTree)
    print('\n****LRN')
    PostorderTraversalBinaryTree(BTree)
    print('\n****LayerTraversal')
    TraversalByLayer(BTree, length)
