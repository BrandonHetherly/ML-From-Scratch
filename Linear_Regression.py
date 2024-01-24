import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class linear_regression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
   
    def fit(self, X, y):
        # parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradent decent
        for _ in range(self.n_iters):
            # apox
            y_predicted = np.dot(X, self.weights) + self.bias

            # dir/gradent of cost function
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

def mse(y, y_hat):
   return np.mean((y - y_hat)**2)

def r_squared(actual_values, predicted_values):
    mean_actual = np.mean(actual_values)

    # Total Sum of Squares (TSS)
    tss = np.sum((actual_values - mean_actual)**2)
    # Residual Sum of Squares (RSS)
    rss = np.sum((actual_values - predicted_values)**2)

    r_squared = 1 - (rss / tss)
    return r_squared


## generate data ##
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# plot the data
figure = plt.figure(figsize= (8,6))
plt.scatter(X, y)
# plt.show()

## test ##
reg = linear_regression(lr=.01)
reg.fit(X=X_train, y=y_train)
predicted_val = reg.predict(X_test)
mse_value = mse(y_test, predicted_val)
print(f"MSE: {round(mse_value,3)}")
print(f"R-squared: {round(r_squared(y_test, predicted_val), 3)}")


#### Compair using Sklearn ####
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"Sklearn MSE: {round(mse,3)}")
print(f"Sklearn R-squared: {round(r2_score(y_test, predictions),3)}")