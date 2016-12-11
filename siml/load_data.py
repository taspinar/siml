import pandas as pd
import os

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
  

def amazon_reviews():
    datafolder = '../datasets/amazon/'
    files = os.listdir(datafolder)
    Y_train, Y_test, X_train, X_test,  = [], [], [], []
    for file in files:
        f = open(datafolder + file, 'rb')
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
