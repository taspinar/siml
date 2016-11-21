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
 
#the training of the naive bayes classifier for text analytics is much easier, 
#we can simply supply a list of categories, and a list of lists containing words
#X = [['this', 'is', 'document','one'], ['this', 'is', 'document', 'two'], ['this', 'is', 'document', 'three']]
#Y = ['label1', 'label2', 'label3']
def train_naive_bayes_text(X,Y):
  class_probabilities = calculate_relative_occurences(Y)
  classes = class_probabilities.keys()
  nb_dict = {}
  for class_label in classes:
    nb_dict[class_label] = {'class_prob': class_probabilities[class_label], 'feature_prob': []}#[]#defaultdict(list)
  for ii in range(0,len(Y)):
    class_label = Y[ii]
    nb_dict[class_label]['feature_prob'] += X[ii]
  #transform the list with all occurences to a dict with relative occurences
  for class_label in classes:
    nb_dict[class_label]['feature_prob'] = calculate_relative_occurences(nb_dict[class_label]['feature_prob'])
  return nb_dict

def classify_naive_bayes(X_element, nb_dict):
  Y_dict = {}
  for class_label in nb_dict.keys():
    nb_dict_features = nb_dict[class_label]['feature_prob']
    class_probability = nb_dict[class_label]['class_prob']
    for word in X_element:
      if word in nb_dict_features.keys():
        relative_word_occurence = nb_dict_features[word]
        class_probability *= relative_word_occurence
      else:
        class_probability *= 0
    Y_dict[class_label] = class_probability
  #return Y_dict
  return get_max_value_key(Y_dict)

def classify_naive_bayes_batch(X,nb_dict):
  predicted_Y_values = []
  for ii in range(0,len(X)):
    X_elem = X[ii]
    predicted_Y = classify_naive_bayes(X_elem, nb_dict)
    predicted_Y_values.append(predicted_Y)
  return predicted_Y_values
