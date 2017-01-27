import pandas as pd
import os
import numpy as np
import random
from io import open

###
def strip_quotations_newline(text):
    text = text.rstrip()
    if text[0] == '"':
        text = text[1:]
    if text[-1] == '"':
        text = text[:-1]
    return text

def expand_around_chars(text, characters):
    for char in characters:
        text = text.replace(char, " "+char+" ")
    return text

def split_text(text):
    text = strip_quotations_newline(text)
    text = expand_around_chars(text, '".,()[]{}:;')
    splitted_text = text.split(" ")
    cleaned_text = [x for x in splitted_text if len(x)>1]
    text_lowercase = [x.lower() for x in cleaned_text]
    return text_lowercase

###
def pow10(x):
    i = 1;
    while((i * 10) < x):
        i *= 10.0;
    return i
    
def normalize_col(col1, method):
    cc_mean = np.mean(col1)
    if method == 'pow10':         
        return col1 / pow10(np.max(col1))
    else:
        return col1 - cc_mean
    
def normalize_matrix(X, method = 'mean'):
    no_rows, no_cols = np.shape(X)
    X_normalized = np.zeros(shape=(no_rows, no_cols))
    X_normalized[:,0] = X[:,0]
    for ii in range(1,no_cols):
        X_normalized[:, ii] = normalize_col(X[:, ii], method)
    return X_normalized    

###
def amazon_reviews():
    datafolder = '../datasets/amazon/'
    files = os.listdir(datafolder)
    Y_train, Y_test, X_train, X_test,  = [], [], [], []
    for file in files:
        f = open(datafolder + file, 'r', encoding="utf8")
        label = file
        lines = f.readlines()
        no_lines = len(lines)
        no_training_examples = int(0.7*no_lines)
        for line in lines[:no_training_examples]:
            Y_train.append(label)
            X_train.append(split_text(line))
        for line in lines[no_training_examples:]:
            Y_test.append(label)
            X_test.append(split_text(line))
        f.close()
    return X_train, Y_train, X_test, Y_test

def adult():
    datafile = '../datasets/adult/adult.data'
    file_test = '../datasets/adult/adult.test'
    df = pd.read_csv(datafile, header=None)
    Y_train = df[14].values
    del df[14]
    del df[2]
    X_train = df.values

    df_test = pd.read_csv(file_test, header=None)
    Y_test = df_test[14].values
    del df_test[14]
    del df_test[2]
    X_test = df_test.values
    return X_train, Y_train, X_test, Y_test

def myopia():
    datafile = '../datasets/myopia/myopia.dat'
    with open(datafile, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        start_col = 2
        no_lines = len(lines)
        no_training_examples = int(0.7*no_lines)
        no_test_examples = no_lines - no_training_examples
        no_features = len(lines[0].split())-start_col
        X, Y = np.zeros(shape=(no_lines, no_features)), np.zeros(shape=no_lines)
        X[:,0] = 1
        rownum = 0
        for line in lines:
            line = line.split()
            line = map(float, line)
            Y[rownum] = int(line[2])
            X[rownum, 1:16] = line[3:18]
            rownum+=1
        X_norm = normalize_matrix(X, 'pow10')
        X_train = X_norm[0:no_training_examples,:]
        Y_train = Y[0:no_training_examples]
        X_test = X_norm[no_training_examples:no_lines,:]
        Y_test = Y[no_training_examples:no_lines]
    return X_train, Y_train, X_test, Y_test

def iris():
    datafile = '../datasets/iris/iris.data'
    df = pd.read_csv(datafile, header=None)
    df_train = df.sample(frac=0.7)
    df_test = df.loc[~df.index.isin(df_train.index)]
    X_train = df_train.values[:,0:4].astype(float)
    Y_train = df_train.values[:,4]
    X_test = df_test.values[:,0:4].astype(float)
    Y_test = df_test.values[:,4]
    return X_train, Y_train, X_test, Y_test
