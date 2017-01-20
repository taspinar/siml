import numpy as np
import load_data as ld
from evaluators import *

class LogisticRegression():
    """
    Class for performing logistic regression.
    """
    def __init__(self, learning_rate = 0.7, max_iter = 1000):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.theta = []
        self.no_examples = 0
        self.no_features = 0
        self.X = None
        self.Y = None
        
    def add_bias_col(self, X):
        bias_col = np.ones((X.shape[0], 1))
        return np.concatenate([bias_col, X], axis=1)
              
    def hypothesis(self, X):
        return 1 / (1 + np.exp(-1.0 * np.dot(X, self.theta)))

    def cost_function(self):
        """
        We will use the binary cross entropy as the cost function. https://en.wikipedia.org/wiki/Cross_entropy
        """
        predicted_Y_values = self.hypothesis(self.X)
        cost = (-1.0/self.no_examples) * np.sum(self.Y * np.log(predicted_Y_values) + (1 - self.Y) * (np.log(1-predicted_Y_values)))
        return cost
        
    def gradient(self):
        predicted_Y_values = self.hypothesis(self.X)
        grad = (-1.0/self.no_examples) * np.dot((self.Y-predicted_Y_values), self.X)
        return grad
        
    def gradient_descent(self):
        for iter in range(1,self.max_iter):
            cost = self.cost_function()
            delta = self.gradient()
            self.theta = self.theta - self.learning_rate * delta
            print("iteration %s : cost %s " % (iter, cost))
        
    def train(self, X, Y):
        self.X = self.add_bias_col(X)
        self.Y = Y
        self.no_examples, self.no_features = np.shape(X)
        self.theta = np.ones(self.no_features + 1)
        self.gradient_descent()
  
    def classify(self, X):
        X = self.add_bias_col(X)
        predicted_Y = self.hypothesis(X)
        predicted_Y_binary = np.round(predicted_Y)
        return predicted_Y_binary

to_bin_y = { 1: { 'Iris-setosa': 1, 'Iris-versicolor': 0, 'Iris-virginica': 0 },
             2: { 'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 0 },
             3: { 'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 1 }
             }

X_train, y_train, X_test, y_test = ld.iris()

Y_train = np.array([to_bin_y[3][x] for x in y_train])
Y_test = np.array([to_bin_y[3][x] for x in y_test])

print("training Logistic Regression Classifier")
lr = LogisticRegression()
lr.train(X_train, Y_train)
print("trained")
predicted_Y_test = lr.classify(X_test)
f1 = f1_score(predicted_Y_test, Y_test, 1)
print("F1-score on the test-set for class %s is: %s" % (1, f1))

# from sklearn.linear_model import LogisticRegression
# logistic = LogisticRegression()
# logistic.fit(X_train,Y_train)
# predicted_Y_test = logistic.predict(X_test)
# f1 = f1_score(predicted_Y_test, Y_test, 1)
# print("F1-score on the test-set for class %s is: %s" % (1, f1))
