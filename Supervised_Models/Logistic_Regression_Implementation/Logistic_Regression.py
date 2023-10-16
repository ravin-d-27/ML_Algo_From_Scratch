import numpy as np

class Logistic_Regression:

    def __init__(self, lr = 0.001, n_iters = 1000) -> None:
        self.lr = 0.001 # learning rate
        self.n_iters = n_iters # number of iterations
        self.weights = None # weights
        self.bias = None # bias 

    def train(self, X_train, y_train):
        # Initialize parameters

        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features) # initialize weights 
        self.bias = 0 # initialize bias

        # Gradient Descent
        for i in range(self.n_iters):
            linear_model = np.dot(X_train, self.weights) + self.bias # y = mx + b (linear model)
            y_predicted = 1/(1 + np.exp(-linear_model)) # sigmoid function

            # update weights and bias

            dw = 1/n_samples * np.dot(X_train.T, (y_predicted - y_train))
            db = i/n_samples * np.sum(y_predicted - y_train)

            self.weights-=self.lr*dw
            self.bias-=self.lr*db


    def predict(self, X_test):
        linear_model = np.dot(X_test, self.weights) + self.bias
        y_predicted = 1/(1 + np.exp(-linear_model)) # sigmoid function
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]

        return y_predicted_cls