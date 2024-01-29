Each file contains an implementation of the model using pure Python and NumPy. Additionally, it includes a test with either a built-in dataset or a created dataset. Lastly, there is an implementation of the model using Sklearn to compare the results and assess any differences, if present. 

### Linear Regression
- temp

### Logistic Regression
- I use gradient descent for optimization
- Accuracy scores differ between models:
  - Sklearns model does not use gradient descent
  - Sklearn defaults to L2 regularization (Ridge), which is not present in the class (I will add regularization in the future)

### KNN
- temp

### Naive Bayes
- Assumes all features are mutually independent
- Assumes all features follow a Gaussian distribution

### Perceptron
- It is one unit of a Artificial neural Network (ANN)
  - Simulates the behavior of a single neuron
- It is a form of supervised learning and the output is binary 
#### Steps 
1) The 'cell' takes inputs and weights
2) Multiply the inputs and weights and sum to create a weighted sum
3) The weighted sum is fed into a activation function which returns the models output

### Support Vector Machine
Goal: Use a linear model to find a decision boundary (hyperplane) that best separates the data 
- The best hyperplane will be determined based on the distance between it and the data on either side of the plane

Note: The difference between the accuracy scores is because Sklearn uses regularization
  - To make scores equal; add 'C=0.0001' to the svm_model as an argument 
