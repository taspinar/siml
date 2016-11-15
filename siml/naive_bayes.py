from collections import Counter, defaultdict
from sets import Set

def calculate_relative_occurences(list1):
  no_examples = len(list1)
  ro = dict(Counter(list1))
  for key in ro.keys():
    ro[key] = ro[key] / float(no_examples)
  return ro

def get_max_value_key(d1):
  values = d1.values()
  keys = d1.keys()
  max_value_index = values.index(max(values))
  max_key = keys[max_value_index]
  return max_key
 
def train_naive_bayes(X,Y):
  classes = list(Set(Y))
  no_features = len(X[0,:])
  nb_dict = {}
  #initialize the naive bayes dictionary with defauldicts
  for class_label in classes:
    nb_dict[class_label] = {}
    for ii in range(0,no_features):
      nb_dict[class_label][ii] = defaultdict(list)
  #fill the dictionary with the values of each feature in the appropiate class
  for ii in range(0,len(Y)):
    class_label = Y[ii]
    for jj in range(0,no_features):
      nb_dict[class_label][jj].append(X[ii,jj])
  #transform the list with all occurences to a dict with relative occurences
  for class_label in classes:
    for ii in range(0,len(no_features)):
      nb_dict[class_label][ii] = calculate_relative_occurences(nb_dict[class_label][ii])
  return nb_dict
  
def classify_naive_bayes_element(X_element, nb_dict):
  Y_dict = {}
  for class_label in nb_dict.keys():
    class_probability = 0 #or initialize it with the actual class probability
    for ii in range(0,len(X_element)):
      relative_feature_values = nb_dict[class_label][ii]
      if X_element[ii] in relative_feature_values.keys():
        class_probability *= relative_feature_values[X_element[ii]]
      else:
        class_probability *= 0
    Y_dict[class_label] = class_probability
  #return Y_dict
  return get_max_value_key(Y_dict)

def classify_naive_bayes_batch(X,Y,nb_dict):
  predicted_Y_values = []
  for ii in range(Y):
    X_elem = X[ii,:]
    predicted_Y = classify_naive_bayes_element(X_elem, nb_dict)
    predicted_Y_values.append(predicted_Y)
  return predicted_Y_values
