import numpy as np

#### Logistic regression from scratch ####
class Logistic_Regression:
    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradent descent
        for _ in range(self.n_iters):
            linear_mod = np.dot(X, self.weights) + self.bias
            y_hat = self._sigmoid(linear_mod)

            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_mod = np.dot(X, self.weights) + self.bias
        y_hat = self._sigmoid(linear_mod)
        y_hat_cls = [1 if i > 0.5 else 0 for i in y_hat]
        return y_hat_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
   
## load data ##
from sklearn.model_selection import train_test_split
from sklearn import datasets

# function to calc accuracy
def accuracy(y, y_hat):
    return np.sum(y == y_hat) / len(y)

bc_data = datasets.load_breast_cancer()
X, y = bc_data.data, bc_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

reg = Logistic_Regression(lr= 0.0001, n_iters=10000)
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