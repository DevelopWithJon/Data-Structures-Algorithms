"""Quick Sort"""

import time

import random

MIN_MERGE = 32
 
def calcMinRun(n):
    """Returns the minimum length of a
    run from 23 - 64 so that
    the len(array)/minrun is less than or
    equal to a power of 2.
 
    e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
    ..., 127=>64, 128=>32, ...
    """
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r
 
 
# This function sorts array from left index to
# to right index which is of size atmost RUN
def insertionSort(arr, left, right):
    for i in range(left + 1, right + 1):
        j = i
        while j > left and arr[j] < arr[j - 1]:
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1
 
 
# Merge function merges the sorted runs
def merge(arr, l, m, r):
     
    # original array is broken in two parts
    # left and right array
    len1, len2 = m - l + 1, r - m
    left, right = [], []
    for i in range(0, len1):
        left.append(arr[l + i])
    for i in range(0, len2):
        right.append(arr[m + 1 + i])
 
    i, j, k = 0, 0, l
     
    # after comparing, we merge those two array
    # in larger sub array
    while i < len1 and j < len2:
        if left[i] <= right[j]:
            arr[k] = left[i]
            i += 1
 
        else:
            arr[k] = right[j]
            j += 1
 
        k += 1
 
    # Copy remaining elements of left, if any
    while i < len1:
        arr[k] = left[i]
        k += 1
        i += 1
 
    # Copy remaining element of right, if any
    while j < len2:
        arr[k] = right[j]
        k += 1
        j += 1
 
 
# Iterative Timsort function to sort the
# array[0...n-1] (similar to merge sort)
def timSort(arr):
    n = len(arr)
    minRun = calcMinRun(n)
     
    # Sort individual subarrays of size RUN
    for start in range(0, n, minRun):
        end = min(start + minRun - 1, n - 1)
        insertionSort(arr, start, end)
 
    # Start merging from size RUN (or 32). It will merge
    # to form size 64, then 128, 256 and so on ....
    size = minRun
    while size < n:
         
        # Pick starting point of left sub array. We
        # are going to merge arr[left..left+size-1]
        # and arr[left+size, left+2*size-1]
        # After every merge, we increase left by 2*size
        for left in range(0, n, 2 * size):
 
            # Find ending point of left sub array
            # mid+1 is starting point of right sub array
            mid = min(n - 1, left + size - 1)
            right = min((left + 2 * size - 1), (n - 1))
 
            # Merge sub array arr[left.....mid] &
            # arr[mid+1....right]
            if mid < right:
                merge(arr, left, mid, right)
 
        size = 2 * size

def quickSort(nums):
    length = len(nums)
    if length<=1:
        return nums
    else:
        pivot = nums.pop()
        
    items_greater = []
    items_lower = []
    
    for item in nums:
        if item>pivot:
            items_greater.append(item)
        else:
            items_lower.append(item)
            
    return quickSort(items_lower) + [pivot] + quickSort(items_greater)

def mergeSort(nums):
    if len(nums) > 1:
        left = nums[:len(nums)//2]
        right = nums[len(nums)//2:]
        
        
        
        #recusion
        mergeSort(left)
        mergeSort(right)
        
        #merge
        i = 0 # left indx
        j = 0 # right indx
        index = 0
        while i<len(left) and j<len(right):
            if left[i] < right[j]:
                nums[index] = left[i]
                i+=1
            else:
                nums[index] = right[j]
                j+=1
            index+=1
        while i<len(left):
            nums[index] = left[i]
            i+=1
            index+=1
        while j<len(right):
            nums[index] = right[j]
            j+=1
            index+=1
    
    return nums
            

def insertSort(nums):
    indexRange = range(1,len(nums))
    for i in indexRange:
        valueToSort = nums[i]
        
        while nums[i-1] > valueToSort and i>0:
            nums[i],nums[i-1]=nums[i-1],nums[i]
            i-=1
    return nums


def bubbleSort(nums):
    
    indexRange = len(nums)-1
    sorted = False
    
    while not sorted:
        sorted=True
        for i in range(0,indexRange):
            if nums[i]>nums[i+1]:
                sorted=False
                nums[i],nums[i+1]=nums[i+1],nums[i]
    return nums
    
    
def selectionSort(nums):
    
    minVal = float('inf')
    indexingLength = len(nums)-1
    for i in indexingLength:
        minVal = i
        
        for j in range(i+1, len(nums)):
            if nums[j] < nums[minVal]:
                minVal = j
        if minVal!=i:
            nums[minVal], nums[i] = nums[i], nums[minVal]
    
    return nums
        
        
    
def test(func):
    start = time.time()
    arr = [random.randint(0,2**20) for _ in range(100000)]
    func(arr)
    print(f"opertation took: {time.time()-start} seconds")

print(mergeSort([34,6,23,7,45,23]))
    


