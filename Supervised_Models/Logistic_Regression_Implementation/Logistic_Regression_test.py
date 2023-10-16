from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Loading the Dataset
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target

# Splitting the Dataset into training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Training the Model
from Logistic_Regression import Logistic_Regression
model = Logistic_Regression()
model.train(X_train, y_train)

preds = model.predict(X_test)

print(preds)

print(accuracy_score(y_test, preds)*100,"%")