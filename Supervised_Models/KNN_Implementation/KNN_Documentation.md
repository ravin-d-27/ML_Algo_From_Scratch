<h1>Knn Class Documentation</h1>

<h2>euclideanDistance</h2>
<p>Calculate the Euclidean distance between two points.</p>
<pre>
def euclideanDistance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
</pre>

<h2>Knn Class</h2>

<h3>__init__(self, k=3)</h3>
<p>Initialize the Knn class.</p>
<pre>
def __init__(self, k=3):
    self.k = k
</pre>

<h3>train(self, X, y)</h3>
<p>Train the Knn model with input data and labels.</p>
<pre>
def train(self, X, y):
    """
    :param X: Input data
    :param y: Labels
    """
    self.X_train = X
    self.y_train = y
</pre>

<h3>predict(self, X)</h3>
<p>Predict the labels for a given set of input data.</p>
<pre>
def predict(self, X):
    """
    :param X: Input data to predict labels for
    :return: Predicted labels
    """
    pred = [self._predict(x) for x in X]
    return np.array(pred)
</pre>

<h3>_predict(self, x)</h3>
<p>Private method to predict the label for a single input.</p>
<pre>
def _predict(self, x):
    """
    :param x: Input data to predict label for
    :return: Predicted label
    """
    # Step 1: Compute the Distances
    distances = [euclideanDistance(x, training) for training in self.X_train]

    # Step 2: Get K Nearest Neighbours and labels
    k_index = np.argsort(distances)[:self.k]
    k_labels = [self.y_train[i] for i in k_index]

    # Step 3: Majority Vote, most common class label
    most_common = Counter(k_labels).most_common(1)

    return most_common[0][0]
</pre>

</body>