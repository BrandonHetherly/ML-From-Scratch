import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

class LinearRegressionModel:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None
        self.intercept = None

    def train(self, X, y):
        # Initializing parameters
        num_samples, num_features = X.shape
        self.coefficients = np.zeros(num_features)
        self.intercept = 0

        # Performing gradient descent
        for _ in range(self.iterations):
            # Predicted values
            predictions = np.dot(X, self.coefficients) + self.intercept

            # Gradients of loss function
            coef_gradient = (1 / num_samples) * np.dot(X.T, (predictions - y))
            intercept_gradient = (1 / num_samples) * np.sum(predictions - y)

            # Updating parameters
            self.coefficients -= self.learning_rate * coef_gradient
            self.intercept -= self.learning_rate * intercept_gradient

    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept

def mean_squared_error(actual, predicted):
    return np.mean((actual - predicted) ** 2)

def calculate_r_squared(actual, predicted):
    actual_mean = np.mean(actual)

    # Calculating TSS and RSS
    total_sum_of_squares = np.sum((actual - actual_mean) ** 2)
    residual_sum_of_squares = np.sum((actual - predicted) ** 2)

    return 1 - (residual_sum_of_squares / total_sum_of_squares)


## generate data ##
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# plot the data
figure = plt.figure(figsize= (8,6))
plt.scatter(X, y)
# plt.show()

## test ##
reg = LinearRegressionModel(learning_rate=.01)
reg.train(X=X_train, y=y_train)
predicted_val = reg.predict(X_test)
mse_value = mean_squared_error(y_test, predicted_val)
print(f"MSE: {round(mse_value,3)}")
print(f"R-squared: {round(calculate_r_squared(y_test, predicted_val), 3)}")


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