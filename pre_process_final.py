## Imports ##
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors as kv
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import numpy as np
import pickle as pkl
import  nltk, csv, os, re, sys, math

from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from keras import backend as K
from keras.models import Sequential
from keras.layers import GRU, Dense, Masking, Dropout, Activation
from keras.callbacks import Callback,EarlyStopping, ModelCheckpoint,CSVLogger
from keras.optimizers import RMSprop

import tensorflow as tf

## Move to working directory
os.chdir('C:/Users/Archie Wiranata/PycharmProjects/Modelling')

def google_vocabs():
    file = open('google_vocab_text.txt', 'r', encoding='utf-8-sig')
    vocab_list = file.read().split(sep='","')
    df_vocab = pd.DataFrame({'original':vocab_list})
    df_vocab['processed'] = df_vocab['original'].str.replace(r'[^\w\s]+', '')
    return df_vocab

## Import and preprocess data
def tfidf (filename):
    name  ='db/' + filename + '.csv'
    df= pd.read_csv(name, encoding = 'ISO-8859-1')

    ## Import stopwords
    nltk.download(['stopwords', 'wordnet'])
    stopwords = nltk.corpus.stopwords.words('english')

    ## Split test and train
    train = df['Date']< '2019-02-11'
    test = df['Date']>= '2019-02-11'

    ## Convert to numpy and use tfidf to vecotrize
    x_train_raw = df['topic'].values[train]
    x_test_raw = df['topic'].values[test]

    y_train = df['Label'].values[train]
    y_test = df['Label'].values[test]

    ## Extracting features ##
    feature_extraction = TfidfVectorizer(stop_words=stopwords, lowercase=True)
    x_train = feature_extraction.fit_transform(x_train_raw)
    x_test = feature_extraction.transform(x_test_raw)

    ## Save to disk
    name = 'db/' + filename + '_x_train.np'
    pkl.dump(x_train, open(name, "wb"))

    name = 'db/' + filename + '_x_test.np'
    pkl.dump(x_test, open(name, "wb"))

    name = 'db/' + filename + '_y_train.np'
    pkl.dump(y_train, open(name, "wb"))

    name = 'db/' + filename + '_y_test.np'
    pkl.dump(y_test, open(name, "wb"))

## Load saved data
def load(filename):
    x_train = pkl.load(open('db/'+filename+'_x_train.np', 'rb'))
    x_test = pkl.load(open('db/'+filename+'_x_test.np', 'rb'))
    y_train = pkl.load(open('db/'+filename+'_y_train.np', 'rb'))
    y_test = pkl.load(open('db/'+filename+'_y_test.np', 'rb'))
    return (x_train, x_test, y_train, y_test)

train_dji = 'train_dji'
#tfidf(train_dji)
x_train, x_test, y_train, y_test = load(train_dji)

## Trim data
x_train = x_train[0:16000]
y_train = y_train[0:16000]
l = int(len(y_test)/16)*16
x_test = x_test[0:l]
y_test = y_test[0:l]

## Network structure
epochs = int(sys.argv[-1])
nb_timesteps = 1
nb_classes = 2
nb_features = x_train.shape[1]
output_dim = 1

## cross-validated model parameters
batch_size = 16
dropout = 0.25
activation = 'sigmoid'
nb_hidden = 128
initialization = 'glorot_normal'

## reshaping X to three dimensions
x_train = csr_matrix.toarray(x_train)
x_train = np.resize(x_train, (x_train.shape[0], nb_timesteps, x_train.shape[1]))

x_test = csr_matrix.toarray(x_test)
x_test = np.resize(x_test, (x_test.shape[0], nb_timesteps, x_test.shape[1]))

##  reshape Y to appropriate dimensions
y_train = np.resize(y_train, (y_train.shape[0], output_dim))
y_test = np.resize(y_test, (y_test.shape[0], output_dim))

## Initialize model
model = Sequential()

model.add(Masking(mask_value=0., batch_input_shape=(batch_size, nb_timesteps, nb_features))) # embedding for variable input lengths
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization,
               batch_input_shape=(batch_size, nb_timesteps, nb_features)))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(GRU(nb_hidden, stateful=True, init=initialization))
model.add(Dropout(dropout))
model.add(Dense(output_dim, activation=activation))

# Configure learning process

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Prepare model checkpoints and callbacks
filepath="db/results/weights-{val_acc:.5f}.hdf5"
checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)

class LearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))

# Training
print('Training')
for i in range(1):
    csv_logger = CSVLogger('db/results/training_log.csv', separator=',', append=True)
    print('Epoch', i+1, '/', epochs)
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              verbose=1,
              nb_epoch=1,
              shuffle=False, # turn off shuffle to ensure training data patterns remain sequential
              callbacks=[checkpointer,csv_logger],
              validation_data=(x_test, y_test))
    model.reset_states()

