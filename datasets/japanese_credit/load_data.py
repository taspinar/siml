#This is code includes the logistic regression algorithm for the classification of the japanese credit dataset.
#goto http://ataspinar.com for a detailed explanation of the math behind logistic regression
#goto https://github.com/taspinar/siml for the full code
#It was used during hackathon4 of the Eindhoven Data Science group:  https://www.meetup.com/Eindhoven-Data-Science-Meetup/events/234115346/

import pandas as pd
from sets import Set
import random
import numpy as np

datafile = './japanese_credit.data'

df = pd.read_csv(datafile, header=None)

column_values = list(df.columns.values)
categorical_columns = [0,3,4,5,6,8,9,11,12]
str_cols = [0,1,3,4,5,6,8,9,11,12,13]
int_columns = [10,13,14]
float_columns = [1,2,7]

#first we select only the rows which do not contain any invalid values
for col in str_cols:
  df = df[df[col] != '?']

#columns containing categorical values are expanded to k different columns with binary values (k is number of categories)
for col in categorical_columns:
  col_values = list(Set(df[col].values))
  for col_value in col_values:
    if col_value != '?':
      df.loc[df[col] == col_value, str(col)+'_is_'+col_value] = 1
    
#remove original columns
for col in categorical_columns:
  del df[col]

#rename the column with the label to 'label' and make it integer
df.loc[df[15] == '+', 'label'] = 1
del df[15]

#normalize the columns with integer values by the mean value
for col in int_columns:
  df[col] = df[col].apply(int)
  col_values = list(df[col].values)
  mean = np.mean(map(int,col_values))
  df[col] = df[col].apply(lambda x: x/float(mean))

#normalize the columns with float values by the mean value
for col in float_columns:
  df[col] = df[col].apply(float)
  col_values = list(df[col].values)
  mean = np.mean(map(float,col_values))
  df[col] = df[col].apply(lambda x: x/mean)
  
df = df.fillna(0)

#create a training and a test set
indices = df.index.values
random.shuffle(indices)
no_training_examples = int(0.7*len(indices))
df_training = df.ix[indices[:no_training_examples]]
df_test = df.ix[indices[no_training_examples:]]

#create and fill the Y matrices of the training and test set
Y = df_training['label'].values
Y_test = df_test['label'].values
del df_training['label']
del df_test['label']

#create the X matrices of the training and test set and initialize with zero
no_features = len(df_training.columns.values)
no_test_examples = len(df_test.index.values)
X = np.zeros(shape=(no_training_examples, no_features))
X_test = np.zeros(shape=(no_test_examples,no_features))

#fill the X matrices
col_values = df_training.columns.values
for ii in range(0,len(col_values)):
  col = col_values[ii]
  X[:,ii] = df_training[col].values
  X_test[:,ii] = df_test[col].values
  
