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
data = pd.read_csv('data2.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 20000  # The number of iterations to perform gradient descent

n = len(X) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)

# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y, s=10) 
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

# R squar method for goodness of fit
ss_t=0
ss_r=0
y_mean=np.mean(Y)
for i in range(n):
    Y_pred=m*X[i]+c
    ss_t+=(Y[i]-y_mean)**2
    ss_r+=(Y[i]-Y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)