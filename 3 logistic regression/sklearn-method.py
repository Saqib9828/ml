# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:51:23 2019

@author: M. Saqib
"""
# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

 # load the data from the file
data = pd.read_csv("data.txt")

# X = feature values, all the columns except the last column
X = data.iloc[:, :-1]

# y = target values, last column of the data frame
y = data.iloc[:, -1]

# filter out the applicants that got admitted
admitted = data.loc[y == 1]

# filter out the applicants that din't get admission
not_admitted = data.loc[y == 0]
    



model = LogisticRegression()
model.fit(X, y)
predicted_classes = model.predict(X)
accuracy = accuracy_score(y.values.flatten(),predicted_classes)
parameters = model.coef_
print(accuracy)

# labelled observations 
admitted = X.loc[np.where(y == 0.0)] 
not_admitted = X.loc[np.where(y == 1.0)] 
plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')

# plotting decision boundary 
x1 = np.arange(30, 100, 1) 
x2 = 120-(parameters[0,0] + parameters[0,1]*x1)/parameters[0,1] 
plt.plot(x1, x2, c='red', label='reg line')

plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()