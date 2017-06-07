# List Comprehensions

list1 = [i for i in range(10)]
print(list1)
# print: 0,1,2,3,4,5,6,7,8,9

list1 = [i for i in range(10) if i % 2 == 0]
print(list1)
# print: 0,2,4,6,8

print(abs(-3),complex(1,2),abs(complex(1,2)))
# print: 3, 1+2j, 2.2360(values)

print(all([True,True,True]),all([True,False,True]))
# print: True, False

print(divmod(7,3),divmod(8,3))
# print: (2,1) (2,2)

print(max(1,2,3,4,5),min(1,2,3,4,5))
# print: 5 1

list1= [1,2,3,4,5]
iter1 = iter(list1)
print(next(iter1),next(iter1),next(iter1))
# print: 1 2 3

print(pow(2,6),round(3.1415),round(3.1415,2),sum([1,2,3,4]))
# print: 64 3 3.14 10

for i in range(0,10,3):
    print(i)
# print: 0 3 6 9

a = [1,2,3]
b = [4,5,6]

zip1 = zip(a,b)
print(list(zip1))
# print: (1,4), (2,5), (3,6)
from math import pi
print(pi)
# print: 3.141592653589793

from datetime import datetime

print(datetime.now())
# print: now time

import calendar
print(calendar.prmonth(2017,2))
# print: calender

# Python Image Library

from PIL import Image
from PIL import ImageFilter
#im = Image.open('./123456.jpg')
#imout = im.filter(ImageFilter.SMOOTH_MORE)
# BLUR, CONTOUR, DETAIL, EDGE_ENHANCE, EDGE_ENHANCE_MORE, EMBOSS, FIND_EDGES, SMOOTH, SMOOTH_MORE, SHARPEN
#imout.show()

# matplotlib
from pylab import *
plot([1,2,3,4])
ylabel('some numbers')
show()

import matplotlib.pyplot as plt
plt.plot([1,2,3,4],[1,4,9,16],'ro')
plt.axis([0,6,0,20])
plt.show()

# scatter
import numpy as np

x = np.random.randn(1000)
y = np.random.randn(1000)
plt.scatter(x,y)
plt.show()

x = np.linspace(0,5,10)
y = x**2

fig = plt.figure()
axes1 = fig.add_axes([0.1,0.1,0.8,0.8])
axes2 = fig.add_axes([0.2,0.5,0.4,0.3])

axes1.plot(x,y,'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('title')

axes2.plot(y,x,'g')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('inser title')

fig.show()

# subplots, step, bar

n = np.array([0,1,2,3,4,5])
fig, axes = plt.subplots(nrows=1,ncols=2)
axes[0].step(n,n**2,lw=2)
axes[0].set_title('step')

axes[1].bar(n,n**2,align='center',alpha=0.5)
axes[1].set_title('bar')

plt.show()