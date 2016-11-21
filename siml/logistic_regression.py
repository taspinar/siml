import math
import numpy as np

def to_binary(x_i):
    #this can probably also be done with round()
    return 1 if x_i > 0.5 else 0

def hypothesis(theta, x_i):
    z = np.dot(theta, x_i)
    sigmoid = 1.0 / (1.0 + math.exp(-1.0*z))
    return 0.9999999999999 if sigmoid == 1 else sigmoid

def determine_correct_guesses(X, Y, theta, m):
    determined_Y = [np.dot(theta, X[ii]) for ii in range(m)]
    determined_Y_binary = [to_binary(elem) for elem in determined_Y]
    correct = 0
    for ii in range(0,m):
        if determined_Y_binary[ii] == Y[ii]:
            correct+=1
    return correct
   
def gradient_descent(X, Y, theta, alpha, m, number_of_iterations=1000):
    for iter in range(0,number_of_iterations):
        cost = (-1.0/m)*sum([Y[ii]*math.log(hypothesis(theta, X[ii]))+(1-Y[ii])*math.log(1-hypothesis(theta, X[ii])) for ii in range(m)])
        grad = (-1.0/m)*sum([X[ii]*(Y[ii]-hypothesis(theta, X[ii])) for ii in range(m)])
        theta = theta - alpha * grad
        correct = determine_correct_guesses(X, Y, theta, m)
        print "iteration %s : cost %s : correct_guesses %s / %s" % (iter, cost, correct, len(Y))
    return theta

