
import numpy as np

class Naive_Bayes:

    def train(self, X_train, y_train):
        n_samples, n_features = X_train.shape # number of samples and number of features
        self._classes = np.unique(y_train) # array of unique values

        n_classes = len(self._classes) # number of classes

        # initialize mean, variance, and priors
        self._mean = np.zeros((n_classes, n_features), dtype = np.float64) # mean
        self._var = np.zeros((n_classes, n_features), dtype = np.float64) # variance
        self._priors = np.zeros(n_classes, dtype = np.float64) # priors

        for c in self._classes:
            X_c = X_train[c==y_train]
            self._mean[c,:] = X_c.mean(axis = 0) # mean
            self._var[c,:] = X_c.var(axis = 0) # variance
            self._priors[c] = X_c.shape[0]/float(n_samples) # probability of each class

    
    def _predict(self, X_test):
        posteriors = []

        for i, c in enumerate(self._classes):
            prior = np.log(self._priors[i]) # log of prior
            class_conditional = np.sum(np.log(self._helper(i, X_test))) # log of class conditional

            posterior = prior + class_conditional 
            posteriors.append(posterior)
        
        return self._classes[np.argmax(posteriors)] # return the class with the highest posterior

    def _helper(self, class_index, x):
        mean = self._mean[class_index]
        var = self._var
        num = np.exp(-(x-mean)**2/(2*var)) # numerator
        den = np.sqrt(2*np.pi*var) # denominator
        return num/den

    
    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test] # predict for each sample
        return y_pred



