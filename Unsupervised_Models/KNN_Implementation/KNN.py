import numpy as np
from collections import Counter

def euclideanDistance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class Knn:
    def __init__(self, k = 3) :
        self.k = k

    def train(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        pred = [self._predict(x) for x in X]
        return np.array(pred)

    def _predict(self, x):
        # Step 1: Compute the Distances

        distances = [euclideanDistance(x, training) for training in self.X_train] # get the distances between the test point and all the training points

        # Step 2: Get K Nearest Neighbours and labels
        
        k_index = np.argsort(distances)[:self.k] # get the indices of the k nearest neighbours
        k_labels = [self.y_train[i] for i in k_index] # get the labels of the k nearest neighbours

        # Step 3: Majority Vote, most common class label
        most_common = Counter(k_labels).most_common(1) # get the most common class label

        return most_common[0][0]
