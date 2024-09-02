#### KNN Implementation From Scratch ####
import numpy as np
from collections import Counter

def calc_euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

class KNearestNeighbors:
    def __init__(self, num_neighbors=3):
        self.num_neighbors = num_neighbors

    def fit(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels

    def predict(self, test_data):
        predictions = [self._make_prediction(single_test) for single_test in test_data]
        return np.array(predictions)

    def _make_prediction(self, single_test):
        # Calculate distances between the test instance and each training instance
        all_distances = [calc_euclidean_distance(single_test, train_instance) for train_instance in self.training_data]
        
        # Identify indices of the k closest training instances
        closest_indices = np.argsort(all_distances)[:self.num_neighbors]
        
        # Retrieve the labels for the k nearest neighbors
        closest_labels = [self.training_labels[i] for i in closest_indices]
        
        # Determine the most frequent label among the nearest neighbors
        most_frequent_label = Counter(closest_labels).most_common(1)[0][0]
        return most_frequent_label

#### Testing ####
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

clf = KNearestNeighbors(num_neighbors=3)
clf.fit(training_data=X_train, training_labels=y_train)
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