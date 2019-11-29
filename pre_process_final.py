## Imports ##
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors as kv
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import numpy as np
import pickle as pkl
import  nltk, csv, os, re, sys, math, datetime

from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import GRU, Dense, Masking, Dropout, Activation, RepeatVector
from keras.callbacks import Callback,EarlyStopping, ModelCheckpoint,CSVLogger
from keras.optimizers import RMSprop
from keras.utils import plot_model

import tensorflow as tf

## Move to working directory
os.chdir('C:/Users/Archie Wiranata/PycharmProjects/Modelling')

def google_vocabs():
    file = open('google_vocab_text.txt', 'r', encoding='utf-8-sig')
    vocab_list = file.read().split(sep='","')
    df_vocab = pd.DataFrame({'original':vocab_list})
    df_vocab['processed'] = df_vocab['original'].str.replace(r'[^\w\s]+', '')
    return df_vocab
filename = 'train_dji'
## Import and preprocess data
def tfidf_train (filename, files):
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

    for i in range(files):
        m = i*16000
        n = (i+1)*16000
        name = 'db/serialized' + filename + '_x_train' + str(i+1) + '.np'
        pkl.dump(x_train[m:n], open(name, "wb"))
        name = 'db/serialized' + filename + '_y_train' + str(i+1) + '.np'
        pkl.dump(y_train[m:n], open(name, "wb"))

    name = 'db/serialized' + filename + '_x_train' + '5' + '.np'
    pkl.dump(x_train[files*16000:], open(name, "wb"))
    name = 'db/serialized' + filename + '_y_train' + '5' + '.np'
    pkl.dump(y_train[files*16000:], open(name, "wb"))

    name = 'db/serialized' + filename + '_x_test.np'
    pkl.dump(x_test, open(name, "wb"))

    name = 'db/serialized' + filename + '_y_test.np'
    pkl.dump(y_test, open(name, "wb"))

## Load saved data
def load(filename):
    x_train = pkl.load(open('db/'+filename+'_x_train.np', 'rb'))
    x_test = pkl.load(open('db/'+filename+'_x_test.np', 'rb'))
    y_train = pkl.load(open('db/'+filename+'_y_train.np', 'rb'))
    y_test = pkl.load(open('db/'+filename+'_y_test.np', 'rb'))
    return (x_train, x_test, y_train, y_test)

def GDS_model(train):
    x_train, x_test, y_train, y_test = load(train)

    ## Trim data
    l = int(len(y_train)/16)*16
    x_train = x_train[0:l]
    y_train = y_train[0:l]

    x_train = x_train[0:16]
    y_train = y_train[0:16]

    l = int(len(y_test)/16)*16
    x_test = x_test[0:l]
    y_test = y_test[0:l]

    ## Network structure
    nb_timesteps = 1
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

    model.add(Masking(mask_value=0., batch_input_shape=(batch_size, nb_timesteps, nb_features), name='Mask')) # embedding for variable input lengths
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU01',
                   batch_input_shape=(batch_size, nb_timesteps, nb_features)))
    model.add(Dropout(dropout, name='DO_01'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU02'))
    model.add(Dropout(dropout, name='DO_02'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU03'))
    model.add(Dropout(dropout, name='DO_03'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU04'))
    model.add(Dropout(dropout, name='DO_04'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU05'))
    model.add(Dropout(dropout, name='DO_05'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU06'))
    model.add(Dropout(dropout, name='DO_06'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU07'))
    model.add(Dropout(dropout, name='DO_07'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU08'))
    model.add(Dropout(dropout, name='DO_08'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU09'))
    model.add(Dropout(dropout, name='DO_09'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU10'))
    model.add(Dropout(dropout, name='DO_10'))
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU11'))
    model.add(Dropout(dropout, name='DO_11'))
    model.add(GRU(nb_hidden, stateful=True, init=initialization, name='GRU12'))
    model.add(Dropout(dropout, name='DO_12'))
    model.add(Dense(output_dim, activation=activation, name='Output'))

    # Configure learning process

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Prepare model checkpoints and callbacks
    filepath="db/results/"+ train +"_best_weights.h5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)
    csv_logger = CSVLogger('db/results/training_log.csv', separator=',', append=True)

    # Training
    print('Training')
    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        verbose=1,
                        epochs=1,
                        shuffle=False, # turn off shuffle to ensure training data patterns remain sequential
                        callbacks=[checkpointer,csv_logger],
                        validation_data=(x_test, y_test))

    ## Evaluating on best results
    model.load_weights(filepath=filepath)
    score = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
    score = dict(zip(model.metrics_names, score))

    summary = model.summary()
    model.save('db/results/model_'+ train + '.h5')
    return history, score, summary

## Main
train_files = ['train_dji', 'train_s&p', 'train_wilshere']
train = 'train_gold'
train_files = ['train_gold', 'train_amazon', 'train_oil', 'train_google'] ## Tesla less than 64000


for train in train_files:
    tfidf_train(train, 4)
train = 'train_tesla'
tfidf_train(train, 3)

history, scores, summary = GDS_model(train)
test = 'oil_oil'

def IKPP_model(test):
    ## set time
    time1 = datetime.datetime.today()

    ## Trim data
    x_train, x_test, y_train, y_test = load(test)
    l = int(len(y_train)/16)*16
    x_train = x_train[0:l]
    y_train = y_train[0:l]
    l = int(len(y_test)/16)*16
    x_test = x_test[0:l]
    y_test = y_test[0:l]

    ## Network structure
    nb_timesteps = 1
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

    ## Load model
    IKPP = load_model('db/results/model_' + train + '.h5')
    IKPP.load_weights('db/results/' + train + '_best_weights.h5')

    ## Freeze layers
    for layer in IKPP.layers[:20]:
        layer.trainable = False

    ## Reset weights
    reset = 0
    if reset == 1:
        for layer in IKPP.layers[-6:]:
            layer.reset_states()

    ## Decoder
    decoder = Sequential()
    decoder.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name="Encoder",
                    batch_input_shape=(batch_size, nb_timesteps, nb_features)))
    decoder.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name="Decoder"))
    decoder.add(Dense(IKPP.layers[0].input_shape[2], activation="linear"))
    # plot_model(decoder, 'db/models/decoder.png')

    ## Combine models
    #merged = Sequential()
    #merged.add(decoder)
    #merged.add(IKPP)
    merged = Model(inputs=decoder.input, outputs=IKPP(decoder.output))
    merged.layers[-1].get_input_at(-2)
    merged.layers[-1].get_input_mask_at(-3)
    merged.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Prepare model checkpoints and callbacks
    filepath="db/results/" + test + "_best_weights.h5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)
    csv_logger = CSVLogger('db/results/model_' +  test + '.csv', separator=',', append=True)

    ## Predict un-tuned model
    score_UT = merged.evaluate(x_test, y_test, verbose=1, batch_size=16)
    score_UT = dict(zip(merged.metrics_names, score_UT))

    # Training
    print('Training')
    while (True):
        time2 = datetime.datetime.today()
        history = merged.fit(x_train,
                             y_train,
                             batch_size=batch_size,
                             verbose=1,
                             nb_epoch=1,
                             shuffle=False,  # turn off shuffle to ensure training data patterns remain sequential
                             callbacks=[checkpointer, csv_logger],
                             validation_data=(x_test, y_test))
        time3 = datetime.datetime.today()
        if ((time3 - time2).seconds*2 + (time2-time1).seconds >= 600):
            break

    ## Evaluating on best results
    merged.load_weights(filepath=filepath)
    score = merged.evaluate(x_test, y_test, batch_size=16, verbose=1)
    score = dict(zip(merged.metrics_names, score))

    summary = merged.summary()
    merged.save('db/results/model_'+ test + '.h5')

    return history,


test_files = ['amazon_amazon', 'tesla_tesla', 'google_google',
              'gold_gold', 'oil_oil']

for test in test_files:
    # tfidf(test)
