import numpy as np

class SupportVectorMachine:

    def __init__(self, learning_rate=0.001, regularization_param=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.regularization_param = regularization_param
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Convert labels to -1 and +1
        y_modified = np.where(y <= 0, -1, 1)
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.num_iterations):
            for index, x_i in enumerate(X):
                condition = y_modified[index] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    # Update rule when condition is met
                    self.weights -= self.learning_rate * (2 * self.regularization_param * self.weights)
                    self.bias -= self.learning_rate * 0
                else:
                    # Update rule when condition is not met
                    self.weights -= self.learning_rate * (2 * self.regularization_param * self.weights - np.dot(x_i, y_modified[index]))
                    self.bias -= self.learning_rate * y_modified[index]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)

## Test ##
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y, y_pred):
    acc = np.sum(y == y_pred) / len(y)
    return acc

X, y = datasets.make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=2.05, random_state=2)
y = np.where(y == 0, -1, 1) # ensure all the labels are -1 or 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

SVM_model = SupportVectorMachine()
SVM_model.fit(X_train, y_train)
predicted = SVM_model.predict(X_test)

print(f"Accuracy: {accuracy(y_test, predicted)}")


## Sklearn Implementation ## 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluate 
accuracy = accuracy_score(y_test, y_pred)
print(f"Sklearn Accuracy: {accuracy}")