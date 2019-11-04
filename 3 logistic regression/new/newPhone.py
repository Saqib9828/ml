import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

# import data
churn_df = pd.read_csv("ChurnData.csv")
churn_df.head()

# Data pre-processing and selection
# Lets select some features for the modeling. Also we change the target data 
# type to be integer, as it is a requirement by the skitlearn algorithm:
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

# Lets define X, and y for our dataset:
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]
y = np.asarray(churn_df['churn'])
y [0:5]

# Also, we normalize the dataset:
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# Train/Test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

#Modeling (Logistic Regression with Scikit-learn)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

yhat = LR.predict(X_test)
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

# evaluation
# Now, lets try log loss for evaluation. In logistic regression, the output can be 
# the probability of customer churn is yes (or equals to 1). This probability is a 
# value between 0 and 1. Log loss( Logarithmic loss) measures the performance of a 
# classifier where the predicted output is a probability value between 0 and 1.
from sklearn.metrics import log_loss
print(log_loss(y_test, yhat_prob))
