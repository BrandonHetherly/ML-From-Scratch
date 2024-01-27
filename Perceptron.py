import numpy as np

class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters = 1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Weights and bias 
        self.weights = np.zeros(n_features)
        self.bias = 0

        # make sure all values are zero or one 
        y_ = np.array([1 if i>0 else 0 for i in y])

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                # apply activation function to get the predicted outputs
                y_pred = self.activation_func(linear_output)
                
                # update the weights 
                update = self.lr * (y_[idx] - y_pred) 
                self.weights += update * x_i
                self.bias = update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return y_pred

    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)
    

## Test ##
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y, y_pred):
    acc = np.sum(y == y_pred) / len(y)
    return acc

X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

percep = Perceptron(learning_rate=0.01, n_iters=1000)
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