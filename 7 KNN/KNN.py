#Load the necessary python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#Load the dataset
df = pd.read_csv('diabetes.csv')
print(df.head(5))
X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values

#importing train_test_split
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#Setup arrays to store training and test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
   
    #Fit the model
    knn.fit(X_train, y_train)
   
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
   
    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test)
#Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=7)

#Fit the model
knn.fit(X_train,y_train)

#Get accuracy. Note: In case of classification algorithms score method represents accuracy.
print(knn.score(X_test,y_test))

#import confusion_matrix
from sklearn.metrics import confusion_matrix
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test,y_pred))