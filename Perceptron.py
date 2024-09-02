import numpy as np

class BinaryPerceptron:

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.activation_function = self._step_function
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        num_samples, num_features = X.shape

        # Initialize weights and bias to zeros
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Convert labels to binary (0 or 1)
        y_binary = np.array([1 if label > 0 else 0 for label in y])

        for _ in range(self.num_iterations):
            for index, x_i in enumerate(X):
                # Compute the linear combination of input and weights
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Apply the activation function to the linear output
                prediction = self.activation_function(linear_output)
                
                # Update the weights and bias based on the prediction error
                update = self.learning_rate * (y_binary[index] - prediction)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = self.activation_function(linear_output)
        return predictions

    def _step_function(self, x):
        return np.where(x >= 0, 1, 0)
    

## Test ##
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y, y_pred):
    acc = np.sum(y == y_pred) / len(y)
    return acc

X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

percep = BinaryPerceptron(learning_rate=0.01, num_iterations=1000)
percep.fit(X_train, y_train)
predictions = percep.predict(X_test)

print(f"Accuracy: {round(accuracy(y_test, predictions), 3)}")

## Sklearn Comparison ##
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# perceptron model
perceptron_model = Perceptron()
perceptron_model.fit(X_train, y_train)
y_pred = perceptron_model.predict(X_test)

print(f'Sklearn Accuracy: {round(accuracy_score(y_test, y_pred),3)}')