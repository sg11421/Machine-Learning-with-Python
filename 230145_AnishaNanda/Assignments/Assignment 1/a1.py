import numpy as np
import pandas as pd
from matplotlib import pyplot
import math

'''
#matrices

def isValid(A,B):
    #your code here
    if len(A[0]) == len(B):
        return True
    else:
        return False

def matrix_multiply(A,B):
    #your code here
    a = []
    for x in A:
        a.append(list(x))
    b = []
    for y in B:
        b.append(list(y))
    prod = []
    for i in range(len(a)):
        prod.append([])
        for j in range(len(b[0])):
            s=0
            for k in range(len(a[0])):
                s+=a[i][k]*b[k][j]
            prod[i]+=[s,]
    return np.array(prod)

def matrix_multiply_2(A,B):
    #your code here
    return np.matmul(A,B)

A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

B = np.array([
    [13, 14, 15],
    [16, 17, 18],
    [19, 20, 21]
])


if isValid(A,B):
    print(f"Result using loops:\n {matrix_multiply(A,B)}")
    print(f"Result using numpy:\n {matrix_multiply_2(A,B)}")
else:
    print(f"Matrix multiplication is not valid")

'''

#normalisation

def mean(x):
    # a,b is the size of the matrix
    s = 0
    if type(x[0]) != list:
        a,b = 1, len(x)
        s = sum(x)
    else:
        a,b = len(x[0]), len(x)
        for i in range(b):
            for k in x[i] :
                s+=k
    return s/(a*b)

x = [[2,3,4],[3,1,5]]
y = [1, 2, 3]
z=[1,1,1]


def standard_deviation(x):
    #root of avg of sum of squares of deviation
    ss = 0
    x_ = mean(x)
    if type(x[0]) != list:
        a,b = 1, len(x)
        for i in range(b):
            ss += (x_ - x[i])**2
    else:
        a,b = len(x[0]), len(x)
        for i in range(b):
            for k in x[i] :
                ss +=(x_ - k)**2
    return (ss/(a*b))**(1/2)

print(standard_deviation(x))
print(standard_deviation(y))  # check this
print(standard_deviation(z))


def zscore_normalisation(x):
    # (x-mean)/std
    x_ = mean(x)
    xstd = standard_deviation(x)
    if type(x[0]) != list:
        a,b = 1, len(x)
        y=[]
        for i in range(b):
            y.append(0)
        for i in range(b):
            try:
                y[i] = (x[i] - x_)/xstd
            except(ZeroDivisionError):
                y[i]=0
    else:
        a,b = len(x[0]), len(x)
        y=[]
        dummy = []
        for i in range(a):
            dummy.append(0)
        for _ in range(b):
            y.append(list(dummy))
        for i in range(b):
            for j in range(len(x[i])):
                y[i][j] =(x[i][j] - x_)/xstd
    return y

#x = [[2,3,4],[3,1,5]]
#print(zscore_normalisation(x))
#y = [1, 2, 3]
#print(zscore_normalisation(y))
#print(zscore_normalisation(z))

'''
def zscore_normalisation_2(x):
    #your code here
    if type(x) is 'list':
        x = np.array(x)
        x = (x - x.mean())/x.std()
        a = []
        for k in x:
            a.append(list(x))
        return a
    else : 
        x = (x - x.mean())/x.std()
        return x
'''

def zscore_normalisation_2(x):
    #your code here
    x = np.array(x)
    x = (x - x.mean())/x.std()
    a = []
    try: 
        for k in list(x):
            a.append(list(k))
    except(TypeError):
        a = list(x)
    return a

#given test case
x = [4, 7, 7, 15, 32, 47, 63, 89, 102, 131]
print(zscore_normalisation_2(np.array(x)))

'''
#sigmoid

def sigmoidfn(x):
    g = 1/(1+ math.exp(-x))
    return g

#test case
print(sigmoidfn(2))

def derivative_sig(x):
    g_ = -math.exp(-x)/(1+ math.exp(-x))**2
    return g_

#test case
print(derivative_sig(2))



#pandas

svp = pd.read_csv("superheated_vapor_properties.csv")

print(svp.shape)

print(svp.is_null.sum())




'''





