# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 23:07:11 2019

@author: M. Saqib
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:01:49 2019

@author: M. Saqib
"""

# imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#reading data
data = pd.read_csv('data-genrator/data3.csv')
print(data.shape)
data.head()

#store X and Y
X=data['add rate'].values
Y=data['sales'].values
n=len(X)
# plot
plt.scatter(X,Y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#calculate x^2,x^3...etc.
sum_x=0
sum_x_2=0
sum_x_3=0
sum_x_4=0
sum_xy=0
sum_x_2_y=0
sum_y=0
for i in range(n):
    sum_x += X[i]
    sum_y += Y[i]
    sum_x_2 += (X[i]**2)
    sum_x_3 += (X[i]**3)
    sum_x_4 += (X[i]**4)
    sum_xy += (X[i]*Y[i])
    sum_x_2_y += (X[i]*X[i]*Y[i])
    
#soleve for a,b,c
a=np.array([[n,sum_x,sum_x_2],[sum_x,sum_x_2,sum_x_3],[sum_x_2,sum_x_3,sum_x_4]])
b=np.array([sum_y,sum_xy,sum_x_2_y])

#a=np.array([2,1,1],[1,3,2],[1,0,0])
#b=np.array(4,5,6)
s = np.linalg.solve(a, b)
print(np.allclose(np.dot(a, s), b))
#ploting curve
x_max=np.max(X)
x_min=np.min(X)

x=np.linspace(x_min,x_max,10)
y=s[0]+(x*s[1])+(x*x*s[2])

plt.plot(x,y,color="red",label="regression line")
plt.scatter(X,Y,s=10, label="actual values")
plt.xlabel("Adding Rate")
plt.ylabel("Sale(product/unit)")
plt.legend()
plt.show()


# mean squared error
mse = np.sum((y - Y)**2)

# root mean squared error
# m is the number of training examples
rmse = np.sqrt(mse/n)
print(rmse)

    

