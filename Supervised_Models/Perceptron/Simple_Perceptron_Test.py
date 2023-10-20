import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import datasets
import matplotlib.pyplot as plt

from Simple_Perceptron import perceptron

X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

p = perceptron()
p.train(X_train, y_train)
predictions = p.predict(X_test)

from sklearn.metrics import accuracy_score
print("Perceptron classification accuracy:", accuracy_score(predictions, y_test)*100)

