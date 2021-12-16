"""Hash Map."""

class HashMap():
    
    def __init__(self, loadSize=0.65):
        self.table = [None]*10
        self.size = 0
        self.loadSize = loadSize
        
    def _hashFunction(self, data: tuple):
        h = 0
        index = (data[0]+h) % len(self.table)
        for i in range(len(self.table)):
            if self.table[index] == None:
                self.table[index] = data
                self.size+=1
                return index
            elif self.table[index] and self.table[index][0] == data[0]:
                print("Key already in HasMap")
                break
            else:
                h+=1
                index = (data[0]+h) % len(self.table)
        print("Unable to add")
        
    def resize(self):
        newTable = [None]*(len(self.table)*2)
        for i in len(self.table)-1:
            newTable[i] = self.table[i]
        self.table = newTable
    
    def show(self):
        print(self.table)
    
    
        
    def add(self, data):
        if (self.size +1)/len(self.table)>=self.loadSize:
            self.resize
            
        print(f"data added to index: {self._hashFunction(data)}")
        
        
h = HashMap()
h.add((12,2))
h.add((9,5))    
h.add((2,10))
h.add((1,11))     

h.show()       
        
