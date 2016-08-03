# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 11:36:30 2015

@author: brjohn
"""

"""
This file contains solutions to Leetcode problems
"""

"""
UglyNumber 1
"""

def isUgly(self, num):
    """
    :type num: int
    :rtype: bool
    """
    if num <= 0:
        return False
    
    for i in [2,3,5]:
        while num%i == 0:
            num = (num/i)
    
    if num != 1:
        return False
    else:
        return True

"""
UglyNumber 2
"""

def nthUglyNumber(n):
    """
    :type n: int
    :rtype: int
    """
  
    ugly = [1]
    
    i2,i3,i5 = 0,0,0
    
    while len(ugly)<n :
        u2 = 2*ugly[i2]
        u3 = 3*ugly[i3]
        u5 = 5*ugly[i5]
        
        min_u = min(u2,u3,u5)
        
        if min_u == u2:
            i2+=1
        elif min_u == u3:
            if min_u%2 == 0:
                i3+=1
                continue
            i3+=1
        else:
            if min_u%2 == 0 or min_u%3 == 0:
                i5+=1
                continue
            i5+=1
        
        ugly.append(min_u)
        
    
    return ugly[n-1]     
    
    
"""
First Bad Version (binary search)
"""

def firstBadVersion(n):
    """
    :type n: int
    :rtype: int
    """
    first = 1
    last = n
    k = int((first+last)/2)
    
    while last-first > 1:
        
        if isBadVersion(k):
            last = k
        else:
            first = k
        k = int((first+last)/2)
    
    if isBadVersion(first):
        return first
    else:
        return last

"""
Add Digits (modulo 9 problem)
"""

def addDigits(num):
    """
    :type num: int
    :rtype: int
    """
    if num == 0:
        return num
    out = num%9
    if out == 0:
        return 9
    return out
        
"""
Count Primes (seive)
"""

#Didn't work:
 
def countPrimes_bad(n):
    """
    :type n: int
    :rtype: int
    
    This method ends up being unacceptably slow because 
    it requires so many searches through the list. 
    like, for each j on the inner while loop, we have to
    access beg_list and inspect every elemt to see if it
    matches j. This makes the algo very slow. 
    It is much better to just use flags for this kind of thing.
    since the list is already sorted, we can just turn the 
    flags (0's or 1's) off and on. 
    """
    if n <=2:
        return 0
    
    beg_list = range(3,n,2)
    primes = [2]
    
    while beg_list:
        i = beg_list[0]
        primes.append(i)
        beg_list.remove(i)
#        beg_copy = [k for k in beg_list]
        j = i**2
        while j <n:
            try:
                beg_list.remove(j)
            except:
                pass
            j += i
    return len(primes)

def countPrimes(n):
    """
    :type n: int
    :rtype: int
    Works by eliminating all multiples of primes from  to sqrt(n)
    gets around math.sqrt(n) by using i*i < n
    """
    lst = [0,0]+[1]*(n-2)

    for i in range(2,n):
        if i*i > n:
            break
        if lst[i]:
            j = i**2
            while j<n:
                lst[j] = 0
                j += i
    return sum(lst)

"""
Happy Number
"""
def isHappy(self, n):
    """
    :type n: int
    :rtype: bool
    """
    num = sum([int(i)**2 for i in str(n)])
    seen = [num]
    while num != 1:
        num = sum([int(i)**2 for i in str(num)])
        if num in seen:
            return False
        else:
            seen.append(num)
    return True
    
"""
Longest Common Prefix
"""
def longestCommonPrefix(self, strs):
    """
    :type strs: List[str]
    :rtype: str
    """
    
    if len(strs) == 0:
        return ""
        
    longest = strs[0]
    for i in strs:
        if '' == i:
            return ""
        if longest == i:
            continue
        else:
            for j in range(len(longest),0,-1):
                if longest[0:j] == i[0:j]:
                    longest = longest[0:j]
                    break
            if longest[0] != i[0]:
                return ""
    return longest
    
"""
Two Sum - HASH MAP
"""
"""
NOTE: Hash map (dict in python) is the quickest way to check
if an entry exits in an UNSORTED array. it's obviously at least
O(n) because you need to build the hashmap (dict), but once 
it's build checking for membership is FAST!
"""
def twoSum(nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    num_dict = {}
    i = 1
    for num in nums:
        if num not in num_dict:
            num_dict[num] = [i]
        else:
            num_dict[num].append(i)
        i+=1
    
    for num in nums:
        check = target-num
        if check in num_dict and num != check:
            return sorted([num_dict[check][0], num_dict[num][0]])
        elif check in num_dict and len(num_dict[check]) >1:
            return sorted(num_dict[check])[:2]

#note: The below algo is better because it searches while
#adding entries to the dict.
#apart from the obvious benefit, it eliminates the need
#for lists (since the second orrence will just be picked up)
def twosum2(num,target):
    n=len(num)
    dict={}
    for i in xrange(n):
        x=num[i]
        if target - x in dict:
            return (dict[target-x]+1,i+1)
        dict[x]=i

#Even MORE clever
def twosums3(nums,target):
    buff_dict = {}
    
    if len(nums) <= 1:
        return False
    
    for i in xrange(len(nums)):
        if nums[i] in buff_dict:
            return buff_dict[nums[i]], i+1
        else:
            buff_dict[target - nums[i]] = i+1
            
# Definition for singly-linked list.
class ListNode(object):
     def __init__(self, x):
         self.val = x
         self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        nxt = (l1.val+l2.val)
        nxt_tens = nxt/10
        current_node = ListNode(nxt%10)
        current1,current2 = l1,l2
        node_list = [current_node]
        while current1.next and current2.next:
            current1,current2 = current1.next,current2.next
            nxt = (current1.val+current2.val)
            node_list.append(ListNode((nxt_tens + nxt%10)%10))
            nxt_tens = nxt/10
        for i in range(len(node_list)-1):
            node_list[0].next = node_list[i+1]            
        return node_list[0]
    
#Add Binary
def addBinary(a, b):
    """
    :type a: str
    :type b: str
    :rtype: str
    """
    #int(x, base=...) returns int representation of x, assuming
    #x is written in base ... (can also use the prefix '0b') for
    #binary
    #bin() returns the binary representation '0b10001010101' 
    a = int(a,base=2)
    b = int(b,base=2)
    return bin(a+b)[2:]
    
#Plus one
def plusOne(digits):
    """
    :type digits: List[int]
    :rtype: List[int]
    """
    a = ''
    for i in digits: a+=str(i)
    return [int(i) for i in str(int(a)+1)]

# majorityElement
"""
Note: this approach is 'ok' considering we're looking for an
element with a specific number of occurences. The only real
downside is the memory taken up by the hashtable (dict).
The algo below accomplished the same task without the use
of a hashtable (Moore voting algo)
"""
def majorityElement(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    count_dict = {}
    for num in nums:
        if num in count_dict:
            count_dict[num] +=1
            if count_dict[num] > len(nums)/2:
                return num
        else:
            count_dict[num] = 1
    return sorted(count_dict.iteritems(), key= lambda x: x[1], reverse=True)[0][0]
    
def majorityElement2(nums):
    """
    :type nums: List[int]
    :rtype: int
    """        
    counter = 0
    for num in nums:
        if counter == 0:
            x = num
            counter = 1
        elif num == x:
            counter +=1
        else:
            counter -=1
        print x,counter, num
    return x
    
def selectionSort(nums):
    n = len(nums)
    for pos in xrange(n-1):
        smallest_loc = pos
        for i in xrange(pos+1, n):
            if nums[i] < nums[smallest_loc]:
                smallest_loc = i
        nums[pos], nums[smallest_loc] = nums[smallest_loc], nums[pos]
    return nums
    
def insertionSort(nums):
    n = len(nums)
    for i in xrange(1,n):
        j = i
        while j>0 and nums[j-1]>nums[j]:
            nums[j],nums[j-1] = nums[j-1],nums[j]
            j -= 1
    return nums

def reverseString1(string):
    """
    Note: reversed() returns an iterator, not a list, hence
    the necessity of using join()
    """
    return ''.join(reversed(string)) 

def reverseString2(string):
    return string[::-1]
    
def reverseString3(string):
    """
    Recursive version - the only one of the three that doesn't
    use python-specific functions
    """
    if len(string) <=1:
        return string
        
    return reverseString3(string[1:])+string[0]

def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """
    largest, c, letter_dict = 0,0,{}
    for i in xrange(len(s)):
        if s[i] in letter_dict:
            dist = i-c
            largest = max(dist,largest)
            c = max(c, letter_dict[s[i]]+1)
        letter_dict[s[i]] = i
    return max(largest, len(s)-c)

#Reverse Integer
#Note: this problem had an overflow restriction, which wanted any int that could not 
# be encoded in 32 bits to return 0
def reverse(x):
    """
    :type x: int
    :rtype: int
    """
    if x < 0:
        out = -1*int(str(x)[:0:-1])
        if out < -2**31:
            return 0
    else:
        out = int(str(x)[::-1])
        if out >= 2**31:
            return 0
    return out

def powerOfTwo(n):
    if n <= 0:
        return False
    else:
        return bin(n & n-1)== '0b0'
        
def powerOfTen(n):
    #Doesn't quite work :/
    #try: 1.000000000003
    sci_n = "{:.1e}".format(n)
    if sci_n[:4] == '1.0e':
        return True
    else:
        return False
        
#ZigZag Conversion
def convert(s, numrows):
    if numrows<=1:
       return s
    
    out = ['' for i in range(numrows)]
    row, step = 0,-1
    
    for letter in s:
        out[row] += letter
        if row == 0 or row == numrows-1:
            step = -step
        row += step

    return ''.join(out)             

a = "PAHNAPLSIIGYIR"

from math import log10
def isPalindrome(x):

    if x<0:
        return False
    elif x<10:
        return True

    m = int(log10(x)) 
    for i in range(m/2+1):
        if x/10**(i) %10 != (x/(10**(m-i))) %10:
            return False
    return True
    
def reverseInt(x):
    if x < 0:
        neg = -1
    else:
        neg = 1
    x = neg*x
    rev = 0
    while x:
        rev = 10*rev + x%10
        x /= 10
    return neg*rev    

def isPalindrome2(x):
    if x < 0:
        return False
    return x == reverseInt(x)

import re
"""
Completely ugly
"""
def romanToInt(s):
    """
    :type s: str
    :rtype: int
    """
    out_int = 0
    thousands = re.search('(\A|[^XIC])M+',s)
    hundreds = re.search('(\A|[^XI])C+([^MD]|\Z)',s)
    tens = re.search('(\A|[^I])X+([^MCDL]|\Z)',s)
    ones = re.search('I+\Z', s)
    
    if re.search('CM', s):
        out_int += 900
    elif re.search('XM', s):
        out_int += 990
    elif re.search('IM', s):
        out_int += 999
    if thousands and len(thousands.group())>0:
        out_int += 1000*len(re.search('M+',thousands.group()).group())

    print out_int    
    
    if re.search('CD',s):
        out_int += 400
    elif re.search('XD',s):
        out_int +=390
    elif re.search('ID', s):
        out_int +=399
    elif re.search('D',s):
        out_int +=500
    
    print out_int    
    
    if re.search('XC',s):
        out_int += 90
    elif re.search('IC',s):
        out_int += 99
    if hundreds and len(hundreds.group())>0:
        out_int += 100*len(re.search('C+',hundreds.group()).group())
    
    print out_int
    
    if re.search('XL',s):
        out_int += 40
    elif re.search('IL', s):
        out_int += 59
    elif re.search('L',s):
        out_int +=50
    
    if re.search('IX', s):
        out_int += 9
    if tens and len(tens.group())>0:
        out_int += 10*len(re.search('X+',tens.group()).group())
    
    print out_int
    
    if re.search('IV', s):
        out_int += 4
    elif re.search('V', s):
        out_int +=5
    
    if ones:
        out_int += len(ones.group())
    
    return out_int
    
def romanToInt2(s):
    add_map = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100,
               'D':500, 'M':1000}
    sub_map = {'IV':2, 'IX':2, 'IL':2, 'IC':2, 'ID':2, 'IM':2,
               'XL':20, 'XD':20, 'XC':20, 'XM':20, 'CM':200, 'CD':200}
    int_out = 0
    
    for c in s:
        int_out += add_map[c]
    for key in sub_map.iterkeys():
        if key in s:
            int_out -= sub_map[key]
    return int_out


"""
Perfect Squares
"""
def numSquares(n):
    """
    816 ms
    """
    roots = {0:0}
    
    j = 1
    
    def check(j):
        for key in roots.keys():
            i = 1
            while key + i*i <=n:
                if i*i+key == n:
                    return j
                else:
                    roots[i*i+key] = j
                i+=1
    
    while j < n+1:
        val = check(j)
        if val:
            return val
        else:
            j+=1

def numSquaresBFS(n):
    """
    804 ms
    surprisingly enough, this isn't that much faster than
    the brute force approach above (which is similar logic
    to the breadth first search approach, below)
    A good example of breadth first search
    """
    i = 1
    squares_list = [0]
    while i*i <= n:
        squares_list.append(i*i)
        i+=1
    current_lvl = [0]
    lvl = 1
    
    while True:
        temp_lvl = []
        for node in current_lvl:
            for s in squares_list:
                if node+s == n:
                    return lvl
                elif node+s < n:
                    temp_lvl.append(node+s)
        current_lvl = list(set(temp_lvl))
        lvl+=1

def minPathSum(grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    m = len(grid)
    n = len(grid[0])

    if m==n==1:
        return grid[0][0]    
    
    new_grid = [[-100 for i in xrange(n)] for i in xrange(m)]
    new_grid[-1][-1] = grid[-1][-1]
    print new_grid
    
    for i in range(n-2,-1,-1):
        new_grid[-1][i] = grid[-1][i]+new_grid[-1][i+1]
    print new_grid
    for j in range(m-2,-1,-1):
        new_grid[j][-1] = grid[j][-1]+new_grid[j+1][-1]
    print new_grid
    for j in range(m-2,-1,-1):
        for i in range(n-2,-1,-1):
            new_grid[j][i] = grid[j][i]+min(new_grid[j+1][i], new_grid[j][i+1])
    print new_grid
    return new_grid[0][0]


#It is actually unnecessary to create a second grid, as is done above
# we can merely update the existing grid. This doesn't make the algo
# any faster, but it saves the extra memory cost.       
def minPathSum2( grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    m = len(grid)
    n = len(grid[0])

    if m==n==1:
        return grid[0][0]    
    
    for i in range(n-2,-1,-1):
        grid[-1][i] += grid[-1][i+1]

    for j in range(m-2,-1,-1):
        grid[j][-1] += grid[j+1][-1]

    for j in range(m-2,-1,-1):
        for i in range(n-2,-1,-1):
            grid[j][i] += min(grid[j+1][i], grid[j][i+1])

    return grid[0][0]

def uniquePaths(m, n):
    """
    :type m: int
    :type n: int
    :rtype: int
    """
    
    if m==n==1:
        return 1
    
    grid = [[1]*n for i in xrange(m)]
    
    for i in xrange(m-2,-1,-1):
        for j in xrange(n-2,-1,-1):
            grid[i][j] = grid[i+1][j]+grid[i][j+1]
    
    return grid[0][0]
    
def uniquePathsWithObstacles(obstacleGrid):
    """
    :type obstacleGrid: List[List[int]]
    :rtype: int
    """
    m = len(obstacleGrid)
    n = len(obstacleGrid[0])
    
    if m == n == 1:
        return 1
    
    ans_grid = [[0]*n for i in xrange(m)]
  
    
    for i in xrange(n-1,-1,-1):
        if not obstacleGrid[-1][i]:
            ans_grid[-1][i] = 1
        else:
            break
    for j in xrange(m-1,-1,-1):
        if not obstacleGrid[j][-1]:
            ans_grid[j][-1] = 1
        else:
            break
    
    for i in xrange(m-2,-1,-1):
        for j in xrange(n-2,-1,-1):
            if not obstacleGrid[i][j]:
                ans_grid[i][j] = ans_grid[i+1][j] + ans_grid[i][j+1]
            else:
                ans_grid[i][j] = 0    
    
    return ans_grid[0][0]
    
def karatsuba(int1, int2):
    """
    input: 2 ints
    output: int1*int2
    """

    if int1<10 or int2<10:
        return int1*int2
    
    n = len(str(int1))/2
    
    a = int1/(10**(n))
    b = int1 % 10**n
   
    c = int2/(10**(n))
    d = int2 % 10**n

    ac = karatsuba(a,c)
    bd = karatsuba(b,d)
    mid = karatsuba((a+b), (c+d)) - ac - bd
    
    return ac*10**(2*n) + mid*10**n + bd
    
def listMerge(lst1, lst2):
    i,j = 0,0
    outlist = []
    l1,l2 = len(lst1), len(lst2)
    
    while i<l1 and j <l2:
        if lst1[i] <= lst2[j]:
            outlist.append(lst1[i])
            i +=1
        else:
            outlist.append(lst2[j])
            j+=1
    
    if i >= l1:
        while j < l2:
            outlist.append(lst2[j])
            j+=1
    if j >= l2:
        while i < l1:
            outlist.append(lst1[i])
            i+=1
    return outlist

def mergeSort(lst):
    n = len(lst)
    
    if n <=1:
        return lst
        
    else:
        lst1 = mergeSort(lst[:n/2])
        lst2 = mergeSort(lst[n/2:])
        
        return listMerge(lst1,lst2)
        
        
def threes_fives(n):
    lst = [1]*(n+1)
    total = 0
    
    i,j = 3,5
    
    while i < n:
        total += i*lst[i]
        lst[i]=0
        i += 3
        
    while j < n:
        total += j*lst[j]
        lst[j]=0
        j += 5
        
   
    return total
    
def naive_mm(mat1, mat2):
    n = len(mat1)
    result = [[] for i in xrange(n)]
    
    for i in xrange(n):
        for j in xrange(n):
            tot = 0
            for k in xrange(n):
                tot+=mat1[i][k]*mat2[k][j]
            result[i].append(tot)
    return result
