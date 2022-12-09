import math
from unittest import result
import numpy as np
import sys
import csv
import random
import matplotlib.pyplot as plt
    # caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('C:/Users/21995/Desktop/Computer Science/CSDS 440/csds440-f22-12/src')
import util

a = np.array([[1,2,3,1],[14,5,1,0],[5,6,8,1],[6,43,1,0],[1,5,6,1],[1,7,9,0], [6,4,1,1], [5,3,6,1], [4,5,1,1]])
x = a[:,0:-1]
y = a[:,-1]

b = [1,2,3,4,5,6]
#print(b[0:0]+b[1:], b[0:1], b[-1:])

print(util.cv_split(x,y,3,stratified = True))
"""test_list = [[('best', 1), ('happy', 8)], [('for', 5), ('geeks', 1)]]
  
# printing original list
print("The original list is : " + str(test_list))
  
# initializing Custom eles
cus_eles = [6, 7, 8]
  
# Row-wise element Addition in Tuple Matrix
# Using enumerate() + list comprehension
result = [[('Gfg', 3), ('is', 3)]]
for idx, val in enumerate(test_list):
    result = np.append(result, [val], axis=0)
# printing result 
print(result)"""


"""
def Q10(N):
    result_x = np.empty((N,2), dtype=float)
    for i in range(0, N):
        result_x[i][0] = round(random.uniform(-1,1),2)
        result_x[i][1] = round(random.uniform(-1,1),2)
    result_y = np.sign(0.5*result_x[:,0]+0.5*result_x[:,1])
    print(result_x)
    print(result_y)
    
Q10(50)

plt.plot()"""