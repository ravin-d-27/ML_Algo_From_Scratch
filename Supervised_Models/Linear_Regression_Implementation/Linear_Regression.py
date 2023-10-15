import numpy as np

class Linear_Regression:

    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def train(self, X, y):
        # We need to initialize the weights

        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features) # initialize the weights to 0
        self.bias = 0 # initialize the bias to 0

        # Gradient Descent
        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias # y = mx + c
            dw = 1/n_samples * np.dot(X.T, (y_predicted - y)) # derivative of weights
            db = 1/n_samples * np.sum(y_predicted - y) # derivative of bias

            self.weights = self.weights - self.lr * dw # update the weights
            self.bias = self.bias - self.lr * db # update the bias

    def predict(self, X):
        y_predicted = np.dot(X,self.weights)+self.bias # y = mx + c
        return y_predicted # return the predicted values  


    