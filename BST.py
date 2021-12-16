import queue
Q = queue.Queue()
class BinaryTree():
    
    def __init__(self, root):
        self.root = Node(root)
        self.size = 0
    
        
    def levelorder(self):
        Q.put(self.root)
        output = []
        while Q.empty() is False:
            curr = Q.get() 
            if curr:
                output.append(curr.value)
                Q.put(curr.left)
                Q.put(curr.right)
        return output
    
    def printTraversal(self, traversalType: str):
        self.output = []
        if traversalType == "preorder":
            self.preOrder(self.root, self.output)
        elif traversalType == "postorder":
            self.postOrder(self.root, self.output)
        elif traversalType == "inorder":
            self.inOrder(self.root, self.output)
        else:
            raise ValueError(f"{traversalType} is not a valid traversal Type. Pleasee provide either Traversal Type of : preorder, postorder, inorder")
        return self.output
        
    def preOrder(self, node, output):
        
        if node == None:
            return
        
        self.output.append(node.value)
        self.preOrder(node.left, output)
        self.preOrder(node.right, output)
    
    def postOrder(self, node, output):
        
        if node == None:
            return
        
        self.postOrder(node.left, output)
        self.postOrder(node.right, output)
        self.output.append(node.value)
    
    def inOrder(self, node, output):
        
        if node == None:
            return
        self.inOrder(node.left, output)
        output.append(node.value)
        self.inOrder(node.right, output)
        
    def add(self, data):
        if self.root == None:
            self.root == Node(data)
        else:
            self._rAdd(self.root, data)
        self.size+=1
 
            
    def _rAdd(self, node, data):
        
        if data < node.value:
            if node.left == None:
                node.left = Node(data)
            else:
                self._rAdd(node.left, data)
        if data > node.value:
            if node.right == None:
                node.right = Node(data)
            else:
                self._rAdd(node.right, data)
            
    def remove(self, data):
        self.dummy = Node(-1)
        self.root = self.rRemove(self.root, data, self.dummy)
        return self.dummy.value
    
    def rRemove(self, node, data, dummy):
        if node == None:
            return "data not found"
        
        elif data < node.value:
            node.left = self.rRemove(node.left, data, dummy)
        elif data > node.value:
            node.right = self.rRemove(node.right, data, dummy)
        
        else:
            dummy.value = node.value
            self.size-=1
            
            if node.left == None and node.right == None:
                return None
            
            elif node.left and node.right:
                dummy2 = Node(-1)
                tempRight = self.removeSuccesor(node.right, dummy2)
                if tempRight:
                    node.right = tempRight
                node.value = dummy2.value
                
            elif node.left:
                return node.left
            elif node.right:
                return node.right
            
        return node
        
    
    def removeSuccesor(self, node, dummy):
        if node.left == None:
            dummy.value = node.value
            if node.right:
                return node.right
        else:
            node.left = self.removeSuccesor(node.left, dummy)
        
        
class Node(BinaryTree):
    
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        
        
tree = BinaryTree("TN")

tree.root.left = Node("MU")
tree.root.right = Node("HZ")
tree.root.left.right = Node("RO")
tree.root.left.right.left = Node("AA")
tree.root.right.left = Node("JE")
tree.root.right.right = Node("DD")
tree.root.right.right.left = Node("GB")
tree.root.right.left.right = Node("SM")
print(tree.printTraversal("preorder"))














