import numpy as np
class Metrics:

    def __init__(self) -> None:
        pass

    def accuracy(self, preds, test):
        return np.sum(preds == test)/len(test)