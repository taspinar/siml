# Synopsis

This repository contains popular Machine Learning algorithms, which have been introduced in various blog posts (http://ataspinar.com). Most of the algorithms are accompanied with blog-posts in which I try to explain the mathematics behind and the interpretation of these algorithms. 


# Motivation
Machine Learning is fun! But more importantly, Machine Learning is easy. 
But the academic literature or even (wikipedia-pages) is full with unnecessary complicated terminology, notation and formulae. This gives people the idea that these ML algorithms can only be understood with a full understanding of advanced math and statistics. Stripped from all of these superfluous language we are left with simple maths which can be expressed in a few lines of code. 

# Notebooks explaining the mathematics
I have also provided some notebooks, explaining the mathematics of some Machine Learning algorithms. 
+ [Linear Regression and Logistic Regression](https://github.com/taspinar/siml/blob/master/notebooks/Linear%20Regression%2C%20Logistic%20Regression.ipynb)

# Installation
To install **siML**:
```python
(sudo) pip install siml
```

or you can clone the repository and in the folder containing setup.py
```python
python setup.py install
```


# Code Example
Once it has been installed, the logistic regression algorithm can be used as follows:

```python
from siml import classifiers

alpha = 0.5
theta = np.ones(no_features)
theta = classifiers.gradient_descent(X, Y, theta, alpha, no_training_examples)
```

'theta' is a vector which now contains the parameter values, calculated with the training set X,Y. 
This vector can be used to classify new examples in the test-set X_test, Y_test (with the dot-product):

```python
determined_Y_test = [np.dot(theta, X_test[ii]) for ii in range(no_test_examples)]
determined_Y_test_binary = [classifiers.to_binary(elem) for elem in determined_Y_test]
```

The determined Y values can be compared against the actual Y-values with the F1-score:
```python
f1 = evaluators.f1_score(determined_Y_test_binary, Y_test)
print "\nf1-score on the test-set is %s" % (f1)
```
