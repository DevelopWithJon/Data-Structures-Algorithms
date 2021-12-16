class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        self.head = None
        
    def full_list(self):
        self.full_list = []
        self.printval = self.head
        while self.printval:
            self.full_list.append(self.printval.val)
            self.printval = self.printval.next
        return self.full_list
        
temp = ListNode()
temp.head = ListNode("Mary")
temp.head.next = ListNode("Andrew")
temp.head.next.next = ListNode("Stephanie")
temp.head.next.next.next = ListNode("Ivan")
temp.head.next.next.next.next = ListNode("Miguel")
temp.head.next.next.next.next.next = ListNode("Andrianna")

print(temp.head.next.val)

print(temp.full_list())

def mystery(node):
    if node==None:
        print("CS 1332 is Cool!")
        return
    
    if node.next!=None and len(node.val) >5:
        mystery(node.next.next)
        print(len(node.val))
    elif len(node.val) %2==0:
        print(node.val)
        mystery(node.next)

print(mystery(temp.head))


    
"""class Solution(object):
    def searchInsert(self, nums, target):
        
        l = 0
        u = len(nums)-1
        
        
        
        
        while l <= u:
            mid = (l+u) //2
            
            if nums[mid] == target:
                pos = mid
                return pos
            else:
                if nums[mid] < target:
                    l = mid+1
                    
                else:
                    u = mid-1
                    
        print(nums[mid])
        
        if nums[mid] > target:
            
            return mid
        elif  nums[mid] < target:
            return mid+1
    
sol = Solution().searchInsert([1,2,3,4,6,9], 8)
print(sol)"""