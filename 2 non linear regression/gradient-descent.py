# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 01:22:37 2019

@author: M. Saqib
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 01:46:45 2019

@author: M. Saqib
"""
# Making the imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing Input data
data = pd.read_csv('data-genrator/data3.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
a = 0
b = 0
c = 0
L = 0.0003  # The learning Rate
epochs =   35000  # The number of iterations to perform gradient descent

n = len(X) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = a+(b*X)+(c*(X**2))  # The current predicted value of Y
    D_a = (-2/n) * sum(Y - Y_pred)  # Derivative wrt a
    D_b = (-2/n) * sum(X*(Y - Y_pred))  # Derivative wrt b
    D_c = (-2/n) * sum((X**2)*(Y - Y_pred))  # Derivative wrt c
    a = a - L * D_a  # Update a
    b = b - L * D_b  # Update b
    c = c - L * D_c  # Update c
    
print (a,b,c)

#ploting curve
x_max=np.max(X)
x_min=np.min(X)

x=np.linspace(x_min,x_max,n)
y=a+(x*b)+(x*x*c)

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


