# -*- coding: utf-8 -*-
""" Min Heap Tree."""

class Heap():
    def __init__(self):
        self.root = None
        self.size = 0
        self.backingArray = [None]
    
    def add(self, data):
        if self.root == None:
            self.size+=1
            self.root = data
            self.backingArray.append(data)
            
        else:
            self.size+=1
            self.backingArray.append(data)
            self._Upheap(data)
    
    def _swap(self, i, j):
        self.backingArray[i], self.backingArray[j] = self.backingArray[j], self.backingArray[i]
            
    def _Upheap(self, data):
        i = self.size
        print(self.backingArray[i])
        print(self.backingArray[i//2])
        while self.backingArray[i//2] != None and self.backingArray[i] < self.backingArray[i//2]:
            self._swap(i, i//2)
            i = i//2
            if i <= 1:
                i = 1
            


heap = Heap()
heap.add(4)
heap.add(3)
heap.add(1)
heap.add(0)

print(heap.backingArray)
