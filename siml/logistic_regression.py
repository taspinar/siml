import math
import numpy as np
import load_data as ld
from evaluators import *

class RegressionBaseClass():
    """
    Class for performing logistic regression.
    """
    def to_binary(self, x_i):
        #this can probably also be done with round()
        return 1 if x_i > 0.5 else 0

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))
        
    def hypothesis(self, x_i):
        z = np.dot(self.theta, x_i)
        return self.sigmoid(z)

    def determine_correct_guesses(self, X, Y, m):
        determined_Y = [np.dot(self.theta, X[ii]) for ii in range(m)]
        determined_Y_binary = [self.to_binary(elem) for elem in determined_Y]
        correct = 0
        for ii in range(0,m):
            if determined_Y_binary[ii] == Y[ii]:
                correct+=1
        return correct

    def gradient_descent(self, X, Y, alpha, number_of_iterations):
        no_rows, no_cols = np.shape(X)
        self.theta = np.ones(no_cols)
        for iter in range(0,number_of_iterations):
            cost = (-1.0/no_rows)*sum([Y[ii]*math.log(self.hypothesis(X[ii]))+(1-Y[ii])*math.log(1-self.hypothesis(X[ii])) for ii in range(no_rows)])
            grad = (-1.0/no_rows)*sum([X[ii]*(Y[ii]-self.hypothesis(X[ii])) for ii in range(no_rows)])
            self.theta = self.theta - alpha * grad
            correct = self.determine_correct_guesses(X, Y, no_rows)
            print "iteration %s : cost %s : correct_guesses %s / %s" % (iter, cost, correct, len(Y))
        
class LogisticRegression(RegressionBaseClass):
    def train(self, X, Y, alpha = 0.0007, number_of_iterations = 1000):
        self.gradient_descent(X, Y, alpha, number_of_iterations)

    # def classify_single_elem(self, X_elem):
        # dp = np.dot(self.theta, X_elem)
        # return self.to_binary(dp)
    
    def classify(self, X):
        #self.determined_Y_values = []
        no_rows, no_cols = np.shape(X)
        determined_Y = [np.dot(self.theta, X[ii,:]) for ii in range(no_rows)]
        determined_Y_binary = [self.to_binary(elem) for elem in determined_Y]
        # print no_rows, no_cols
        # for ii in range(0,no_rows):
            # X_elem = X[ii,:]
            # print len(X_elem)
            # prediction = self.classify_single_elem(X_elem)
            # print prediction
            # self.determined_Y_values.append(prediction)
        return determined_Y_binary

X_train, Y_train, X_test, Y_test = ld.myopia()
print("training Logistic Regression Classifier")
lr = LogisticRegression()
lr.train(X_train, Y_train, 0.6, 1000)
print("trained")
predicted_Y = lr.classify(X_train)
f1 = f1_score(predicted_Y, Y_train, 1)
print("F1-score on the test-set for class %s is: %s" % (1, f1))
