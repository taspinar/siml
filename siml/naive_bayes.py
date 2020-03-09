from collections import Counter, defaultdict
from evaluators import *
import load_data as ld
import numpy as np

class NaiveBaseClass:
    def calculate_relative_occurences(self, list1):
        no_examples = len(list1)
        ro_dict = dict(Counter(list1))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(no_examples)
        return ro_dict

    def get_max_value_key(self, d1):
        values = list(d1.values())
        keys = list(d1.keys())
        max_value_index = values.index(max(values))
        max_key = keys[max_value_index]
        return max_key
        
    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)

class NaiveBayes(NaiveBaseClass):
    """
    Naive Bayes Classifier:
    It is trained with a 2D-array X (dimensions m,n) and a 1D array Y (dimension 1,n).
    X should have one column per feature (total m) and one row per training example (total n).
    After training a dictionary is filled with the class probabilities per feature.
    """
    def train(self, X, Y):
        self.labels = np.unique(Y)
        no_rows, no_cols = np.shape(X)
        self.initialize_nb_dict()
        self.class_probabilities = self.calculate_relative_occurences(Y)
        #fill self.nb_dict with the feature values per class
        for label in self.labels:
            row_indices = np.where(Y == label)[0]
            X_ = X[row_indices, :]
            no_rows_, no_cols_ = np.shape(X_)
            for jj in range(0,no_cols_):
                self.nb_dict[label][jj] += list(X_[:,jj])
        #transform the dict which contains lists with all feature values 
        #to a dict with relative feature value occurences per class
        for label in self.labels:
            for jj in range(0,no_cols):
                self.nb_dict[label][jj] = self.calculate_relative_occurences(self.nb_dict[label][jj])

    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            for ii in range(0,len(X_elem)):
              relative_feature_values = self.nb_dict[label][ii]
              if X_elem[ii] in relative_feature_values.keys():
                class_probability *= relative_feature_values[X_elem[ii]]
              else:
                class_probability *= 0
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)
                    
    def classify(self, X):
        self.predicted_Y_values = []
        no_rows, no_cols = np.shape(X)
        for ii in range(0,no_rows):
            X_elem = X[ii,:]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)
        return self.predicted_Y_values

            
class NaiveBayesText(NaiveBaseClass):
    """"
    When the goal is classifying text, it is better to give the input X in the form of a list of lists containing words.
    X = [
    ['this', 'is', 'a',...],
    (...)
    ]
    Y still is a 1D array / list containing the labels of each entry
    """
    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = []
            
    def train(self, X, Y):
        self.class_probabilities = self.calculate_relative_occurences(Y)
        self.labels = np.unique(Y)
        self.no_examples = len(Y)
        self.initialize_nb_dict()
        for ii in range(0,len(Y)):
            label = Y[ii]
            self.nb_dict[label] += X[ii]
        #transform the list with all occurences to a dict with relative occurences
        for label in self.labels:
            self.nb_dict[label] = self.calculate_relative_occurences(self.nb_dict[label])
                
    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            nb_dict_features = self.nb_dict[label]
            for word in X_elem:
                if word in nb_dict_features.keys():
                    relative_word_occurence = nb_dict_features[word]
                    class_probability *= relative_word_occurence
                else:
                    class_probability *= 0
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)

    def classify(self, X):
        self.predicted_Y_values = []
        n = len(X)
        for ii in range(0,n):
            X_elem = X[ii]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)  
        return self.predicted_Y_values

####


X_train, Y_train, X_test, Y_test = ld.amazon_reviews()
print("training naive bayes")
nbc = NaiveBayesText()
nbc.train(X_train, Y_train)
print("trained")
predicted_Y = nbc.classify(X_test[:100])
y_labels = np.unique(Y_test)
for y_label in y_labels:
    f1 = f1_score(predicted_Y, Y_test, y_label)
    print("F1-score on the test-set for class %s is: %s" % (y_label, f1))

    
X_train, Y_train, X_test, Y_test = ld.adult()
print("training naive bayes")
nbc = NaiveBayes()
nbc.train(X_train, Y_train)
print("trained")
predicted_Y = nbc.classify(X_test)
y_labels = np.unique(Y_test)
for y_label in y_labels:
    f1 = f1_score(predicted_Y, Y_test, y_label)
    print("F1-score on the test-set for class %s is: %s" % (y_label, f1))
