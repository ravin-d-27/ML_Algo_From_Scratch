import numpy as np

class perceptron:

    def __init__(self, learning_rate = 0.001, n_iters = 1000) -> None:
        self.lr = learning_rate
        self.n_iters = n_iters
        self.act_func = self._unit_step_func
        self.weight = None
        self.bias = None

    def train(self, X_train, y_train):
        
        n_samples, n_features = X_train.shape

        # Initialize values
        self.weight = np.zeros(n_features)
        self.bias = 0

        y_conv = [1 if i>0 else 0 for i in y_train]
        y_conv = np.array(y_conv)

        for i in range(self.n_iters):
            for index, j in enumerate(X_train): 
                linear_output = np.dot(j,self.weight)+self.bias
                y_pred = self.act_func(linear_output)

                updates = self.lr*(y_conv[index]-y_pred)
                self.weight = self.weight + (updates * j)
                self.bias = self.bias + (updates)


    def predict(self, X_test):
        
        linear_model = np.dot(X_test, self.weight) + self.bias
        y_pred = self.act_func(linear_model)
        return y_pred


    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)

