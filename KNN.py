#### KNN Implementation From Scratch ####
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
   
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
   
    def _predict(self, x):
        # Find distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # get k-nearest samples
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote (most common label)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


#### Testing ####
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

clf = KNN(k=3)
clf.fit(X=X_train, y=y_train)
predictions = clf.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy}")


#### Sklearn Comparison ####
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Create classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
# fit and predict
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)

# Accuracy of sklearn model
accuracy = accuracy_score(y_test, y_pred)
print(f"Sklearn Accuracy: {accuracy}")
