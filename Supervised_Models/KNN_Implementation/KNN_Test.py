from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
X,y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=0)

print(X_train.shape, y_train.shape)
print(X_train[0], y_train[0])

# Using the User defined KNN Model

from KNN import Knn

model = Knn(k=4)
model.train(X_train, y_train)

preds = model.predict(X_test)

print(preds)

# Using the User defined metrics class

from metrics import Metrics
obj = Metrics()
print(obj.accuracy(preds,y_test))