## Imports ##
import tensorflow as tf
import pickle as pkl
import pandas as pd
import numpy as np
import os, sys

import nltk
from keras.preprocessing.text import Tokenizer
from keras import backend as K
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Activation
from keras.callbacks import Callback,EarlyStopping, ModelCheckpoint,CSVLogger
from keras.optimizers import RMSprop

## Move to working directory
os.chdir('C:/Users/Archie Wiranata/PycharmProjects/Modelling')

## Pre-process ##

train_dji = 'train_dji'

def init_tokenize(filename): ## Load and tokenize master data
    name = 'db/' + filename + '.csv'
    df = pd.read_csv(name, encoding='ISO-8859-1')

    train = df['Date'] < '2019-02-11'
    test = df['Date'] >= '2019-02-15'

    nltk.download(['stopwords', 'wordnet'])
    stopwords = nltk.corpus.stopwords.words('english')
    processed_topic = []
    for sentences in df.topic:
        sentence = [word for word in sentences if word not in stopwords]
        processed_topic.append(sentence)
    df["processed"] = processed_topic
    t = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', )
    t.fit_on_texts(df.processed[train])

    x_train = t.texts_to_matrix(df["processed"][train], mode='count')
    y_train = df["Label"][train]
    x_test = t.texts_to_matrix(df["processed"][test], mode='count')
    y_test = df["Label"][test]
    return t, x_train, y_train, x_test, y_test

token, x_train, y_train, x_test, y_test = init_tokenize(train_dji)

## Trim data
x_train = x_train[0:16000]
y_train = y_train[0:16000]
l = int(len(y_test)/16)*16
x_test = x_test[0:l]
y_test = y_test[0:l]

## Initialize attention model
model_attn = Sequential()


