<h1>Linear Regression Class Documentation</h1>

<h2>Description</h2>
<p>
    This is a simple implementation of Linear Regression using Python's NumPy library.
</p>

<h2>Methods</h2>

<h3>__init__(lr=0.001, n_iters=1000)</h3>
<p>
    Initializes a Linear_Regression object.
</p>
<p>
    <strong>Parameters:</strong><br>
    <code>lr</code> (float, optional): Learning rate for gradient descent. Default is 0.001.<br>
    <code>n_iters</code> (int, optional): Number of iterations for gradient descent. Default is 1000.
</p>

<h3>train(X, y)</h3>
<p>
    Trains the linear regression model on the given data.
</p>
<p>
    <strong>Parameters:</strong><br>
    <code>X</code> (numpy.ndarray): Input features with shape (n_samples, n_features).<br>
    <code>y</code> (numpy.ndarray): Target values with shape (n_samples,).
</p>

<h3>predict(X)</h3>
<p>
    Predicts target values for input features.
</p>
<p>
    <strong>Parameters:</strong><br>
    <code>X</code> (numpy.ndarray): Input features with shape (n_samples, n_features).
</p>
<p>
    <strong>Returns:</strong><br>
    <code>y_predicted</code> (numpy.ndarray): Predicted target values with shape (n_samples,).
</p>

<h2>Examples</h2>
<pre>
# Example usage
model = Linear_Regression(lr=0.01, n_iters=2000)
model.train(X, y)
y_pred = model.predict(X_test)
</pre>
