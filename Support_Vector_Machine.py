import numpy as np

class SVM:

    def __init__(self, lr=0.001, lambda_p=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_p= lambda_p
        self.n_iters = n_iters
        self.wt = None
        self.bias = None

    # fit the training samples and labels
    def fit(self, X, y):
        # make sure all lables are -1 or plus 1
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        # weight and bias
        self.wt = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.wt) - self.bias) >= 1
                if condition:
                    # update if condition is True
                    self.wt -= self.lr * (2 * self.lambda_p * self.wt)
                    self.bias = self.lr * 0
                else:
                    # update if condition is False
                    self.wt -= self.lr * (2 * self.lambda_p * self.wt - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def predict(self, X):
        linear_model = np.dot(X, self.wt) - self.bias
        return np.sign(linear_model)

## Test ##
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y, y_pred):
    acc = np.sum(y == y_pred) / len(y)
    return acc

X, y = datasets.make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.05, random_state=12)
y = np.where(y == 0, -1, 1) # ensure all the labels are -1 or 1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

SVM_model = SVM()
SVM_model.fit(X_train, y_train)
predicted = SVM_model.predict(X_test)

print(f"Accuracy: {accuracy(y_test, predicted)}")


## Sklearn Implementation ## 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluate 
accuracy = accuracy_score(y_test, y_pred)
print(f"Sklearn Accuracy: {accuracy}")