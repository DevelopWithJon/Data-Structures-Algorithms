"""Linked Lists"""

#Singly Linked Without tail

"""class SLinkedList():
    
    def __init__(self):
        self.head = None
    
        
    def removeFromFront(self):
        print(self.head.val)
        self.head = self.head.next
        
        
    def removeFromBack(self):
        if self.head == None: return None
        elif self.head.next == None:
            self.head = None
        
        curr = self.head 
        while curr.next.next:
            curr = curr.next
        curr.next = None
        
    def addToFront(self, val):
        newNode = Node(val)
        if self.head:
            newNode.next = self.head
        self.head = newNode
    
    def addToBack(self, val):
        newNode = Node(val)
        if self.head is None:
            self.head = newNode
            return
        curr = self.head
        while curr.next!= None:
            curr = curr.next
        curr.next = newNode
            
    def getSize(self):
        if not self.head: return 0
        count = 0
        curr = self.head
        while curr:
            count+=1
            curr = curr.next
        return count
    
    def show(self):
        showList = []
        if not self.head:
            return None
        curr = self.head
        showList.append("head->" + str(curr.val))
        while curr.next:
            curr = curr.next
            showList.append(curr.val)
        return showList
    
class Node():
    
    def __init__(self, val):
        self.val = val
        self.next = None"""

#Singly Tailed Linked List

"""class STLinkedList():
    
    def __init__(self):
        self.head = None
        self.tail = None
    
        
    def removeFromFront(self):
        if self.head.next:
            self.head = self.head.next
        else:
            self.head = None
            self.tail = None
        
        
    def removeFromBack(self):
        if self.head == None: return None
        elif self.head.next == None:
            self.head = self.tail = None
        
        curr = self.head 
        while curr.next.next:
            curr = curr.next
        curr.next = None
        
    def addToFront(self, val):
        newNode = Node(val)
        if self.head:
            newNode.next = self.head
        self.head = newNode
    
    def addToBack(self, val):
        newNode = Node(val)
        if self.head is None:
            self.head = self.tail = newNode
            return
        self.tail.next = newNode
        self.tail = self.tail.next
            
    def getSize(self):
        if not self.head: return 0
        count = 0
        curr = self.head
        while curr:
            count+=1
            curr = curr.next
        return count
    
    def show(self):
        showList = []
        if not self.head:
            return None
        curr = self.head
        showList.append("head->" + str(curr.val))
        while curr.next:
            curr = curr.next
            showList.append(curr.val)
        return showList
    
class Node():
    
    def __init__(self, val):
        self.val = val
        self.next = None"""

# Doubly Linked List

    


        