def reverseString(word):
    if not word:
        return ""
    
    return reverseString(word[1:]) + word[0]


def isPalindrome(word):
    
    if len(word) <2:
        return True
    
    if word[0].lower() == word[-1].lower():
        return isPalindrome(word[1:-1])
    
    return False

def decToBin(num, result=""):
    
    if num ==0:
        return result
    div, remainder = divmod(num,2)
    result+=str(remainder)
    return decToBin(div, result)
    
def sumofNatural(num):
    if num<=1:
        return num
    return num + sumofNatural(num-1)

def binarySearch(A, left, right, x):
    
    if left>right:
        return (left+right)//2 + 1
    
    mid = (left+right)//2
    
    if A[mid] > x:
        return binarySearch(A, left, mid-1, x)
    elif A[mid] < x:
        return binarySearch(A, mid+1, right, x)
    else:
        return mid
    
arr = [1,3,53,345,2356,4039]
print(binarySearch(arr, 0, len(arr)-1, 4049))
        
def mergeSort(nums,start,end):
    
    if start==end:
        mid = (start+end)//2
        mergeSort(nums,start,mid)
        mergeSort(nums, mid+1, end)
        merge(nums, start, mid, end)
        
def merge(nums, start, mid, end):
    temp = [0]* len(nums)
    
    i,j,k=start,mid,0
    
    while i<=mid and j<=end:
        if nums[i] <= nums[j]:
            temp[k] = nums[j]
            k+=1
            j+=1
        else:
            temp[k] = nums[i]
            i+=1
            k+=1
    while i<=mid:
        temp[k] = nums[i]
        k+=1
        i+=1
        
    while j<=end:
        temp[k] = nums[j]
        k+=1
        j+=1
    
    for i in range(start,end):
        nums[i] = temp[i-start]

class ListNode():
    def __init__(self,val):
        self.val = val
        self.head = None
        self.next = None
    
head = ListNode(3)
head.next = ListNode(4)
