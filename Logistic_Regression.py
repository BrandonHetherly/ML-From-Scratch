import numpy as np

#### Logistic regression from scratch ####
class LogisticRegression:
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        # Initialize parameters
        num_samples, num_features = X.shape
        self.coefficients = np.zeros(num_features)
        self.intercept = 0

        # Perform gradient descent
        for _ in range(self.num_iterations):
            linear_output = np.dot(X, self.coefficients) + self.intercept
            predictions = self._sigmoid(linear_output)

            # Compute gradients
            gradient_weights = (1 / num_samples) * np.dot(X.T, (predictions - y))
            gradient_intercept = (1 / num_samples) * np.sum(predictions - y)

            # Update parameters
            self.coefficients -= self.learning_rate * gradient_weights
            self.intercept -= self.learning_rate * gradient_intercept

    def predict(self, X):
        linear_output = np.dot(X, self.coefficients) + self.intercept
        predictions = self._sigmoid(linear_output)
        class_labels = [1 if prob > 0.5 else 0 for prob in predictions]
        return class_labels

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
   
## load data ##
from sklearn.model_selection import train_test_split
from sklearn import datasets

# function to calc accuracy
def accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)

bc_data = datasets.load_breast_cancer()
X, y = bc_data.data, bc_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

reg = LogisticRegression(learning_rate = 0.0001, num_iterations = 10000)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
print(f"Accuracy: {round(accuracy(y_test, predictions),3)}")


### Compair using Sklearn ###
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# logistic regression model
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

# Evaluate
print(f"Sklearn Accuracy: {round(accuracy_score(y_test, y_pred),3)}")