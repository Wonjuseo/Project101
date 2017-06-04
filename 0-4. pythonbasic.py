# python basic

print(1+3,50-4,5000/3)
#4,46,1666.666666666667
watch = 10000
print(watch)
# 10000
a = 'pig'
b = 'dad'
print(a+b,a+' '+b)
#pigdad, pig dad
family = ['mother','father','gentleman','lady']
print(len(family))
# 4
print(family[3])
# not gentleman, but lady
# mother - 0, father - 1, gentleman - 2, lady -3 
family.remove('gentleman')
print(family)
# mother, father, lady
num = 1
while num <= 100:
    if(num%10 == 0):
        print(num)
    num = num + 1
# 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
print(range(2,7),list(range(2,7)))
# range(2,7), [2,3,4,5,6] 2 ~ 6
for i in range(2,7):
    print(i)
#2 3 4 5 6

def user_add(a,b):
    return a+b

print(user_add(5,6))
# 11

def countdown(n):
    if n==0:
        print("finished")
    else:
        print(n)
        countdown(n-1)

countdown(5)
# 5 4 3 2 1 finished

def function_a(x):
    a = 3
    b = 5
    y = a*x+b
    return y

print(function_a(10))
# 35

def equal_a(a,b):
    return a == b

print(equal_a(3,5),' ',equal_a(3,3))
#False True

x = 'apple'
print(x[0],x[2:4],x[3:])
# a, pl, le

prine = [1,2,3,4,5]
prine.append(6)
print(prine)
# 1,2,3,4,5,6

prine = [10,8,3,11,15]
prine.sort()
print(prine)
# 3,8,10,11,15
del prine[4]
print(prine)
# 3,8,10,11

matrix = [[1,2,3],[4,5,6],[7,8,9]]
print(matrix[1][2])
# 6

characters = []
sentences = 'good'
for char in sentences:
    characters.append(char)
print(characters)
# g o o d

jone = [10, 20, 30]
mike = [30, 20, 10]
smith = [10, 10, 10]

students = [jone,mike,smith]

for scores in students:
    print(scores)
# 10,20,30     30,20,10     10,10,10

for scores in students:
    total = 0
    for s in scores:
        total = total+s
    average = total/3
    print(scores,average)

# 10, 20 ,30 , 20.0      30,20,10 20.0      10,10,10, 10.0


import random
print(random.random())
# random number
print(random.randrange(1,7))# 1~6
abc = ['a','b','c','d','e']
random.shuffle(abc)
print(abc)
# shuffled results
print(random.choice(abc))
# random choice

class food:
    def nu(self):
        return "good"

kimchi = food()
print(kimchi.nu())
# good

class warrior:
    str_ = 20
    dex_ = 25
    vital = 20
    energy = 15

    def attack(self):
        return 'attack'
    
    def train(self):
        self.str_ += 3
        self.dex_ += 3
        self.vital += 3


kim = warrior

print(kim.str_,kim.dex_,kim.vital,kim.energy,kim.attack(kim))

kim.train(kim)

print(kim.str_,kim.dex_,kim.vital,kim.energy,kim.attack(kim))
