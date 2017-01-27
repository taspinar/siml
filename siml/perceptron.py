import numpy as np
import load_data as ld
from evaluators import *
import random

def generate_data(no_points):
    X = np.zeros(shape=(no_points, 2))
    Y = np.zeros(shape=no_points)
    for ii in range(no_points):
        X[ii][0] = random.randint(1,9)+0.5
        X[ii][1] = random.randint(1,9)+0.5
        Y[ii] = 1 if X[ii][0]+X[ii][1] >= 13 else -1
    return X, Y

class Perceptron():
    """
    Class for performing Perceptron.
    X is the input array with n rows (no_examples) and m columns (no_features)
    Y is a vector containing elements which indicate the class 
        (1 for positive class, -1 for negative class)
    w is the weight-vector (m number of elements)
    b is the bias-value
    """
    def __init__(self, b = 0, max_iter = 1000):
        self.max_iter = max_iter
        self.w = []
        self.b = 0
        self.no_examples = 0
        self.no_features = 0
    
    def train(self, X, Y):
        self.no_examples, self.no_features = np.shape(X)
        self.w = np.zeros(self.no_features)
        for ii in range(0, self.max_iter):
            w_updated = False
            for jj in range(0, self.no_examples):
                a = self.b + np.dot(self.w, X[jj])
                if np.sign(Y[jj]*a) != 1:
                    w_updated = True
                    self.w += Y[jj] * X[jj]
                    self.b += Y[jj]
            if not w_updated:
                print("Convergence reached in %i iterations." % ii)
                break
        if w_updated:
            print(
            """
            WARNING: convergence not reached in %i iterations.
            Either dataset is not linearly separable, 
            or max_iter should be increased
            """ % self.max_iter
                )

    def classify_element(self, x_elem):
        return int(np.sign(self.b + np.dot(self.w, x_elem)))
            
    def classify(self, X):
        predicted_Y = []
        for ii in range(np.shape(X)[0]):
            y_elem = self.classify_element(X[ii])
            predicted_Y.append(y_elem)
        return predicted_Y


X, Y = generate_data(100)
p = Perceptron()
p.train(X, Y)
X_test, Y_test = generate_data(50)
predicted_Y_test = p.classify(X_test)
f1 = f1_score(predicted_Y_test, Y_test, 1)
print("F1-score on the test-set for class %s is: %s" % (1, f1))



#####
        
# to_bin_y = { 1: { 'Iris-setosa': 1, 'Iris-versicolor': -1, 'Iris-virginica': -1 },
             # 2: { 'Iris-setosa': -1, 'Iris-versicolor': 1, 'Iris-virginica': -1 },
             # 3: { 'Iris-setosa': -1, 'Iris-versicolor': -1, 'Iris-virginica': 1 }
             # }

# X_train, y_train, X_test, y_test = ld.iris()

# Y_train = np.array([to_bin_y[1][x] for x in y_train])
# Y_test = np.array([to_bin_y[1][x] for x in y_test])

# p = Perceptron()
# print("Training Perceptron Classifier")
# p.train(X_train, Y_train)

# predicted_Y_test = p.classify(X_test)
# f1 = f1_score(predicted_Y_test, Y_test, 1)
# print("F1-score on the test-set for class %s is: %s" % (1, f1))
