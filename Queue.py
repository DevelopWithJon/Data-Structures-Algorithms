"""Queue"""

#Linked List Queue





#Array Queue

class Queue():
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.front = 0
        self.back = 0
        self.Queue = [None]*self.capacity
        
    def enqueue(self, value):
        if self.size < self.capacity:
            self.Queue[self.getBack()] = value
        else:
            raise Exception("Queue is Full")
        self.size+=1
        
    def dequeue(self):
        if self.size == 0:
            raise Exception("Cannot remove from empty Queue")
        self.Queue[self.front] = None
        self.front+=1
        self.size-=1
    
    def getBack(self):
        self.back = (self.front + self.size) % self.capacity
        return self.back
    
    def show(self):
        return self.Queue
    
    def clear(self):
        self.size = 0
        self.Queue = [None]*self.capacity
        
q = Queue(4)
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)
q.enqueue(4)
q.dequeue()
q.enqueue(5)
q.dequeue()
q.dequeue()
q.enqueue(9)

print(q.show())

        