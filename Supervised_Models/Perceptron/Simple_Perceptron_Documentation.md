<h1>Perceptron Class Documentation</h1>
<h2>Class: perceptron</h2>

<p>
    The perceptron class is a simple implementation of a single-layer perceptron, which is a basic type of artificial neural network.
    It is used for binary classification tasks.
</p>
<h3>Constructor: __init__(learning_rate=0.001, n_iters=1000)</h3>

<p>
    Initializes a new instance of the perceptron class with the specified learning rate and number of iterations.
</p>

<table>
    <tr>
        <th>Parameter</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>learning_rate</td>
        <td>The learning rate used for weight updates during training.</td>
    </tr>
    <tr>
        <td>n_iters</td>
        <td>The number of iterations for training.</td>
    </tr>
</table>
<h3>Method: train(X_train, y_train)</h3>

<p>
    Trains the perceptron model with the given training data.
</p>

<table>
    <tr>
        <th>Parameter</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>X_train</td>
        <td>Training data (features).</td>
    </tr>
    <tr>
        <td>y_train</td>
        <td>Target labels for training data.</td>
    </tr>
</table>

<h3>Method: predict(X_test)</h3>

<p>
    Predicts the target labels for the given test data.
</p>

<table>
    <tr>
        <th>Parameter</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>X_test</td>
        <td>Test data (features).</td>
    </tr>
</table>

<h3>Method: _unit_step_func(x)</h3>

<p>
    Applies the unit step function to the input <code>x</code>.
</p>

<table>
    <tr>
        <th>Parameter</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>x</td>
        <td>Input value.</td>
    </tr>
</table>
<h2>Example Usage</h2>

<pre>
import numpy as np

# Instantiate the perceptron
perceptron_model = perceptron(learning_rate=0.01, n_iters=100)

# Train the model
perceptron_model.train(X_train, y_train)

# Predict
y_pred = perceptron_model.predict(X_test)
</pre>
