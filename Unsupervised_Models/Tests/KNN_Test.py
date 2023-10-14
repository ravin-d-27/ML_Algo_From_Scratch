from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
X,y = dataset.data, dataset.target

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=0)

print(X_train.shape, y_train.shape)
print(X_train[0], y_train[0])

