import numpy as np
import itertools
from itertools import repeat

p = np.array([[1,2,3],[4,5,6],[7,8,9]])
#for val in range(np.size(p)/3):
bbb = np.array([row[1] for row in p])
aaa=np.array(list(repeat (bbb, 6)))
ff=np.concatenate((aaa, p.T)).T

x = np.array([[2,2],[5,5],[8,8]])
y = np.array([10,11,12]).T

test = []
for j in range(3):
     for i in range(3):
         test([i][j])