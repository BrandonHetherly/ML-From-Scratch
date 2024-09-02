import numpy as np

class GaussianNaiveBayes:

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.classes = np.unique(y)
        num_classes = len(self.classes)

        # Initialize mean, variance, and prior probabilities
        self.mean = np.zeros((num_classes, num_features), dtype=np.float64)
        self.variance = np.zeros((num_classes, num_features), dtype=np.float64)
        self.prior_probs = np.zeros(num_classes, dtype=np.float64)

        # Calculate mean, variance, and prior probabilities for each class
        for idx, cls in enumerate(self.classes):
            X_cls = X[y == cls]
            self.mean[idx, :] = X_cls.mean(axis=0)
            self.variance[idx, :] = X_cls.var(axis=0)
            self.prior_probs[idx] = X_cls.shape[0] / float(num_samples)

    def predict(self, X):
        predictions = [self._classify(sample) for sample in X]
        return predictions

    def _classify(self, sample):
        posterior_probs = []

        for idx, cls in enumerate(self.classes):
            prior_log = np.log(self.prior_probs[idx])
            class_conditional_log = np.sum(np.log(self._calculate_pdf(idx, sample)))
            posterior = prior_log + class_conditional_log
            posterior_probs.append(posterior)
        
        return self.classes[np.argmax(posterior_probs)]
    
    # Probability dennsity function (PDF) of normal 
    def _calculate_pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        variance = self.variance[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

## testing ##
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y, y_hat):
    accuracy = np.sum(y == y_hat) / len(y)
    return accuracy

# create the data 
X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

naiveBayes = GaussianNaiveBayes()
naiveBayes.fit(X_train, y_train)
predictions = naiveBayes.predict(X_test)
print(f"Accuracy: {round(accuracy(y_test, predictions),3)}")


## Compairing with Sklearn 
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Init the Gaussian Naive Bayes classifier
naive_bayes_classifier = GaussianNB()

naive_bayes_classifier.fit(X_train, y_train)
y_pred = naive_bayes_classifier.predict(X_test)

# performance
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Sklearn Accuracy: {round(accuracy,3)}")