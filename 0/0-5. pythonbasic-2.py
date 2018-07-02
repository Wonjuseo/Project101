# python basic -2
# adding
a = 1
b = 2

c = a+b

print(c)

# if-else sentences
if a > 0:
    print("a>0")
else:
    print("a<0")

# import library
import math

n = math.sqrt(16.0)
print(n)
# data type
print(int(3.5),2e3,float("1.6"),float("inf"),float("-inf"),bool(0),bool(-1),bool("False"))

# imaginary number

v = 2 + 3j
print(v.real,v.imag)

a = [1,2,3,4]
b = 3 in a
print(b)#True

# String
a = "ABC"
b = a
print(a is b)#True

# Print
print("name: %s ages: %d" % ("John",16))
print("number = %0.4f, number2 = %10.5f" % (3.141592,3.141592))

# String
s = "Hello"
print(type(s),s[1])

s = '.'.join(['AB','CD','DF'])
print(s)
s = ' '.join(['AB','CD','EF'])
print(s)

items = 'AB,CD,EF'.split(',')
print(items)

s = "Name:{0}, Age:{1}".format("John",10)
print(s)
s = "Name:{name},Age:{age}".format(name="John",age=10)
print(s)
area = (10,20)
s = "width: {x[0]}, height {x[1]}".format(x = area)
print(s)

# list and for loop
list1 = ["AB","CD","EF"]
for s in list1:
    print(s)

sum = 0
for i in range(101):#0~100
    sum += i
print(sum)

a = []
a = ["AB",10,False]

x = a[1]
a[1] = "Good"
y = a[-1]
print(x,a[1],y)

# Merge

a = [1,2]
b = [3,4,5]

c = a+b
print(c)

d = a*3

print(d)

# list search

list1 = "the john is the good man".split()
a = list1.index('john')# 1
n = list1.count('the')# 2
print(a,n)

# list comprehension

list1 = [n ** 3 for n in range(10) if n % 2 == 0]
print(list1)

# Tuple

name = ("Kim","Park")
print(name)

firstname,lastname=name
print(lastname,',',firstname)

# Dictionary

scores = {"kim":100,"Park":90}
v = scores["kim"]
scores["Park"]= 95
print(scores)
scores["Lee"]=100
del scores["kim"]
print(scores)

for a in scores:
    val = scores[a]
    print("%s : %d" % (a,val))

keys = scores.keys()
for k in keys:
    print(k)

values = scores.values()
for v in values:
    print(v)

scores.update({"Park":100,"Lee":80})
print(scores)

# Set

our_set = {'True','False','True','True'}
s = set(our_set)
print(s)

num_set = {1,2,3,4,5}
num_set.add(10)
print(num_set)
num_set.update({15,20,25})
print(num_set)
num_set.remove(1)
print(num_set)
num_set.clear()
print(num_set)

a = {1,2,3}
b = {3,4,5}

i = a & b
print(i)

u = a | b
print(u)

d = a-b

print(d)

# Class

class rectangle:
    count = 0

    def __init__(self,width,height):
        self.width = width
        self.height = height
        rectangle.count +=1

    def calculation(self):
        area = self.width*self.height
        return area
    def square_s(width,height):
        return width == height

square = rectangle.square_s(5,5)
square_1 = rectangle(5,5)
print(square,square_1.calculation())

# Thread
import threading

def sum(low,high):
    total = 0
    for i in range(low,high):
        total += 1
    print("Subthread,  ",total)

t = threading.Thread(target=sum,args=(1,10000))
t.start()

print("Main Thread   ")
