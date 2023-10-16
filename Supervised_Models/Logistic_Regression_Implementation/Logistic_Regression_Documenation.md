<h1>Logistic Regression Class Documentation</h1>
<h2>Class: Logistic_Regression</h2>
<p>
        This class implements a simple logistic regression model for binary classification tasks.
    </p>

<h3>Constructor</h3>
    <h4>__init__(self, lr=0.001, n_iters=1000)</h4>
    <p>
        Initializes a new instance of the Logistic_Regression class.
    </p>
    <h5>Parameters:</h5>
    <ul>
        <li><strong>lr</strong> (float, optional): Learning rate for gradient descent. Default is 0.001.</li>
        <li><strong>n_iters</strong> (int, optional): Number of iterations for training. Default is 1000.</li>
    </ul>
    <h5>Attributes:</h5>
    <ul>
        <li><strong>lr</strong> (float): Learning rate for gradient descent.</li>
        <li><strong>n_iters</strong> (int): Number of iterations for training.</li>
        <li><strong>weights</strong> (numpy.ndarray): Model weights.</li>
        <li><strong>bias</strong> (float): Model bias.</li>
    </ul>
    <h3>Methods</h3>
    <h4>train(self, X_train, y_train)</h4>
    <p>
        Trains the logistic regression model on the provided training data.
    </p>
    <h5>Parameters:</h5>
    <ul>
        <li><strong>X_train</strong> (numpy.ndarray): Input features for training.</li>
        <li><strong>y_train</strong> (numpy.ndarray): Target labels for training.</li>
    </ul>
    <h4>predict(self, X_test)</h4>
    <p>
        Predicts the target labels for the given test data using the trained model.
    </p>
    <h5>Parameters:</h5>
    <ul>
        <li><strong>X_test</strong> (numpy.ndarray): Input features for testing.</li>
    </ul>
    <h5>Return Value:</h5>
<p>
    Returns a list of predicted labels for the test data.
</p>

</body>

