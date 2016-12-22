import random
import numpy as np

def generate_data(no_points):
    X = np.zeros(shape=(no_points, 2))
    Y = np.zeros(shape=no_points)
    for ii in range(no_points):
        X[ii][0] = random.randint(1,9)+0.5
        X[ii][1] = random.randint(1,9)+0.5
        Y[ii] = 1 if X[ii][0]+X[ii][1] >= 12 else -1
    return X, Y
  

def Perceptron(X, Y, b=0, max_iter=20):
    """
    b is the bias,
    X is the input array with n rows (training examples) and m columns (features)
    """
    n,m = np.shape(X)
    #weight-vector
    w = np.zeros(m)
    for ii in range(0,max_iter):
      for jj in xrange(n):
        x_i = X[jj]
        y_i = Y[jj]
        a = b + np.dot(w, x_i)
        if np.sign(y_i*a) != 1:
          w += y_i*x_i
          b += y_i
          print("iteration %s; new weight_vector: %s - new b: %s" % (ii, w, b))
          
X, Y = generate_data(100)
Perceptron(X, Y, max_iter=20)