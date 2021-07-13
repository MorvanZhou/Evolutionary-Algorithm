import numpy as np

a = np.array([1,1,1])
b = np.array([0,0,1])

# Solution
dist = np.linalg.norm(a-b)
print(dist)

for i in range(5):
   for j in range(i+1,5):
      print('i=',i,'   j=',j)