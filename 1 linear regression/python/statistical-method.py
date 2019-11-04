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
data = pd.read_csv('data2.csv')
print(data.shape)
data.head()

#store X and Y
X=data['add rate'].values
Y=data['sales'].values

# plot
plt.scatter(X,Y,s=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#get mean
x_mean=np.mean(X)
y_mean=np.mean(Y)

#total population size
n=len(X)

#calculate slope
m=0
num=0
dem=0
for i in range(n):
    num=num+(X[i]-x_mean)*(Y[i]-y_mean)
    dem=dem+(X[i]-x_mean)**2
m=num/dem

#calculate cut on y axis C
c=y_mean-(m*x_mean)

print(m,c)

#ploting line
x_max=np.max(X)
x_min=np.min(X)

x=np.linspace(x_min,x_max,1000)
y=m*x+c

plt.plot(x,y,color="red",label="regression line")
plt.scatter(X,Y,s=10, label="actual values")
plt.xlabel("Adding Rate")
plt.ylabel("Sale(product/unit)")
plt.legend()
plt.show()

# R squar method for goodness of fit
ss_t=0
ss_r=0
for i in range(n):
    y_pred=m*X[i]+c
    ss_t+=(Y[i]-y_mean)**2
    ss_r+=(Y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)