 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

 
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
'''
# data format
6.2,2.9,4.3,1.3,Iris-versicolor
5.1,2.5,3.0,1.1,Iris-versicolor
5.7,2.8,4.1,1.3,Iris-versicolor
6.3,3.3,6.0,2.5,Iris-virginica
5.8,2.7,5.1,1.9,Iris-virginica
7.1,3.0,5.9,2.1,Iris-virginica
6.3,2.9,5.6,1.8,Iris-virginica
.
.
.
'''
 
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

 
dataset = pd.read_csv(path, names=headernames)
dataset.head()

 
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(X_train, y_train)

 
y_pred = classifier.predict(X_test)

 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

