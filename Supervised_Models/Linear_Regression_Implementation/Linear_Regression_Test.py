import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

print(X_train.shape)
print(y_train.shape)

from Linear_Regression import Linear_Regression

model = Linear_Regression(lr=0.01)
model.train(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)


# Testing Sklearn's mean_squared_error function
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(mse)

# Testing my MSE function
from metrics import MeanSquaredError
mse = MeanSquaredError()
print(mse.MSE(y_test, y_pred))  

# Testing Sklearn's r2_score function
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

# Testing my r2_score function
from metrics import MeanSquaredError
r2 = MeanSquaredError()
print(r2.r2_score(y_test, y_pred))

