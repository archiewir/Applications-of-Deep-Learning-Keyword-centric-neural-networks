## Imports ##
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle as pkl
import  nltk, csv, os, re, sys, time

from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score

from keras import backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import GRU, Dense, Masking, Dropout, Activation, InputLayer
from keras.callbacks import Callback,EarlyStopping, ModelCheckpoint,CSVLogger
os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/Graphviz2.38/bin"
from keras.utils import plot_model

import tensorflow as tf

## Move to working directory
os.chdir('C:/Users/Archie Wiranata/PycharmProjects/Modelling')

def tfidf_self (filename):
    name  ='db/' + filename + '.csv'
    df= pd.read_csv(name, encoding = 'ISO-8859-1')

    ## Import stopwords
    nltk.download(['stopwords', 'wordnet'])
    stopwords = nltk.corpus.stopwords.words('english')

    ## Convert to numpy and use tfidf to vecotrize
    x_train_raw = df['topic'].values
    y_train = df['Label'].values

    ## Extracting features ##
    feature_extraction = TfidfVectorizer(stop_words=stopwords, lowercase=True)
    x_train = feature_extraction.fit_transform(x_train_raw)

    ## Save to disk
    name = 'db/serialized/' + filename + '_x_test.np'
    pkl.dump(x_train, open(name, "wb"))

    name = 'db/serialized/' + filename + '_y_test.np'
    pkl.dump(y_train, open(name, "wb"))

'''
## Import and preprocess data
references = ['train_dji', 'train_s&p', 'train_wilshere']
filenames = ['amazon_amazon', 'tesla_tesla', 'google_google',
              'gold_gold', 'oil_oil']
'''

#filename = 'train_gold'
def tfidf_train (filename, files):
    name  ='db/' + filename + '.csv'
    df= pd.read_csv(name, encoding = 'ISO-8859-1')
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')

    ## Import stopwords
    nltk.download(['stopwords', 'wordnet'])
    stopwords = nltk.corpus.stopwords.words('english')

    ## Split test and train

    train = df['Date']< pd.datetime(2019,2,10)
    test = df['Date']>= pd.datetime(2019,2,10)

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
        name = 'db/serialized/' + filename + '_x_train' + str(i+1) + '.np'
        pkl.dump(x_train[m:n], open(name, "wb"))
        name = 'db/serialized/' + filename + '_y_train' + str(i+1) + '.np'
        pkl.dump(y_train[m:n], open(name, "wb"))

    name = 'db/serialized/' + filename + '_x_train' + '5' + '.np'
    pkl.dump(x_train[files*16000:], open(name, "wb"))
    name = 'db/serialized/' + filename + '_y_train' + '5' + '.np'
    pkl.dump(y_train[files*16000:], open(name, "wb"))

    name = 'db/serialized/' + filename + '_x_test.np'
    pkl.dump(x_test, open(name, "wb"))

    name = 'db/serialized/' + filename + '_y_test.np'
    pkl.dump(y_test, open(name, "wb"))

def tfidf_asset_classes (reference, filename):
    name ='db/' + reference + '.csv'
    refer = pd.read_csv(name, encoding = 'ISO-8859-1')
    refer['Date'] = pd.to_datetime(refer['Date'], format='%d-%m-%y')
    train = refer['Date'] < pd.datetime(2019, 2, 10)
    refer = refer['topic'].values[train]
    name  ='db/' + filename + '.csv'
    df= pd.read_csv(name, encoding = 'ISO-8859-1')

    ## Import stopwords
    nltk.download(['stopwords', 'wordnet'])
    stopwords = nltk.corpus.stopwords.words('english')

    ## Convert to numpy and use Tf-idf to vectorize
    x= df['topic'].values
    y= df['Label'].values

    ## Extracting features ##
    feature_extraction = TfidfVectorizer(stop_words=stopwords, lowercase=True).fit(refer)
    x = feature_extraction.transform(x)

    name = 'db/serialized/' + filename + '_' + reference + '_x_test.np'
    pkl.dump(x, open(name, "wb"))
    name = 'db/serialized/' + filename + '_' + reference + '_y_test.np'
    pkl.dump(y, open(name, "wb"))

'''
references = ['train_dji', 'train_s&p', 'train_wilshere'] ## Insert train_dji after
references2 = ['train_amazon', 'train_tesla', 'train_google', 'train_gold', 'train_oil']
filenames = ['amazon_amazon', 'tesla_tesla', 'google_google', 'gold_gold', 'oil_oil']

for reference in references:
    for filename in filenames:

for i in range(len(references2)):
    tfidf_asset_classes(references2[i], filenames[i])
'''

## Load saved data
def load_test(filename):
    x_test = pkl.load(open('db/serialized/'+filename+'_x_test.np', 'rb'))
    y_test = pkl.load(open('db/serialized/'+filename+'_y_test.np', 'rb'))
    x_test = csr_matrix.toarray(x_test)
    return (x_test, y_test)

def load_train(filename, i, nb_timesteps, output_dim):
    x_train = pkl.load(open('db/serialized/' + filename + '_x_train' + str(i+1) + '.np', 'rb'))
    y_train = pkl.load(open('db/serialized/' + filename + '_y_train' + str(i+1) + '.np', 'rb'))
    x_train = csr_matrix.toarray(x_train)
    x_train = np.resize(x_train, (x_train.shape[0], nb_timesteps, x_train.shape[1]))
    y_train = np.resize(y_train, (y_train.shape[0], output_dim))
    return x_train, y_train


def GDS_model(train, ep, files) :
    time1 = str(datetime.today().time())

    ## load test data
    x_test, y_test = load_test(train)

    ## Network structure
    nb_timesteps = 1
    output_dim = 1
    nb_features = x_test.shape[1]

    ## cross-validated model parameters
    batch_size = 16
    dropout = 0.25
    activation = 'sigmoid'
    nb_hidden = 128
    initialization = 'glorot_normal'

    ##  reshape Y to appropriate dimensions
    x_test = np.resize(x_test, (x_test.shape[0], nb_timesteps, x_test.shape[1]))
    y_test = np.resize(y_test, (y_test.shape[0], output_dim))

    ## Trim data
    #x_train = x_train[0:160]
    #y_train = y_train[0:160]

    l = int(len(y_test)/16)*16
    x_test = x_test[0:l]
    y_test = y_test[0:l]

    ## Initialize model
    K.clear_session()
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
    date = str(datetime.today().date())
    filepath ="db/results/"+ train +"_all_weights.h5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)
    filepath = "db/results/best_weights/" + train + '_' + "_best_weights.h5"
    best_weights = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True, monitor="val_acc", mode='max')
    csv_logger = CSVLogger('db/results/logs/' + train + '_' + date+ '_training_log.csv', separator=',', append=True)

    # Training
    print('Training')
    epoch = ep
    histories = []
    for e in range(epoch): ## change verbose
        for i in range (files-1): ## 64000
            print('epoch: ' + str(e+1) + '/' + str(epoch))
            print ('section: ' + str(i+1) + '/' + str(files))
            x_train, y_train = load_train(train, i,nb_timesteps, output_dim)
            history = model.fit(x_train, y_train,
                                batch_size=batch_size, verbose=1, epochs=1,
                                shuffle=False, # turn off shuffle to ensure training data patterns remain sequential,
                                callbacks=[checkpointer, csv_logger, best_weights], validation_data=(x_test, y_test))
            histories.append(history.history)
            del x_train, y_train

        print('section: ' + str(files) + '/' + str(files) )
        x_train, y_train = load_train(train, files-1, nb_timesteps, output_dim) ## 565
        l = int(len(y_train) / 16) * 16
        x_train = x_train[0:l]
        y_train = y_train[0:l]
        history = model.fit(x_train, y_train,
                            batch_size=batch_size, verbose=1, epochs=1,
                            shuffle=False,  # turn off shuffle to ensure training data patterns remain sequential,
                            callbacks=[checkpointer, csv_logger, best_weights], validation_data=(x_test, y_test))
        histories.append(history.history)
        model.reset_states()
    model.save('db/results/model_'+ train + '.h5')
    histories = pd.DataFrame(histories)
    histories.to_csv('db/results/history/model_'+ train + '_histories.csv')

    ## Evaluating on best results
    print("Evaluating Best Weights")
    model.load_weights(filepath=filepath)
    score = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
    score = dict(zip(model.metrics_names, score))
    score = pd.DataFrame([score])
    score.to_csv('db/results/best_score/model_'+ train + '_best_score.csv')

    print(time1)
    print(str(datetime.today().time()))
    return histories, score

## Main
train_files = ['train_dji', 'train_s&p', 'train_wilshere']
train_files2 = ['train_gold', 'train_amazon', 'train_oil', 'train_google', 'train_tesla']

#tfidf(train_dji)
#history, scores= GDS_model(train_files[0], 20, 5)
#train_dji = {'history':history, 'score': scores}

#history, scores= GDS_model(train_files[1], 20, 5)
#train_sp = {'history':history, 'score': scores}

#history, scores= GDS_model(train_files[2], 20, 5)
#train_wsi = {'history':history, 'score': scores}

#history, scores= GDS_model(train_files2[0], 10, 5)
#train_gold = {'history':history, 'score': scores}

#history, scores= GDS_model(train_files2[1], 10, 5)
#train_amazon = {'history':history, 'score': scores}

#history, scores= GDS_model(train_files2[2], 10, 5)
#train_oil = {'history':history, 'score': scores}

#history, scores= GDS_model(train_files2[3], 10, 5)
#train_google = {'history':history, 'score': scores}

#history, scores= GDS_model(train_files2[4], 10, 4)
#train_tesla= {'history':history, 'score': scores}

#history, scores= GDS_model(train_files2[0], 10, 5)
#train_gold = {'history':history, 'score': scores}

## Model 1
def GRU12(train, test):
    ## load model
    timeStart = time.time()
    K.clear_session()
    GDS = load_model('db/results/model_' + train + '.h5')
    GDS.load_weights('db/results/best_weights/' + train + '_best_weights.h5')

    ## Set un-trainable layers 4:-6
    for layer in GDS.layers[:-6]:
        layer.trainable = False

    GDS.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

    ## load test files
    name = test + '_' + train
    x_test, y_test = load_test(name)

    ## Network structure
    nb_timesteps = 1
    output_dim = 1
    nb_features = x_test.shape[1]

    ## cross-validated model parameters
    batch_size = 16
    dropout = 0.25
    activation = 'sigmoid'
    nb_hidden = 128
    initialization = 'glorot_normal'

    ##  reshape Y to appropriate dimensions
    x_test = np.resize(x_test, (x_test.shape[0], nb_timesteps, x_test.shape[1]))
    y_test = np.resize(y_test, (y_test.shape[0], output_dim))

    ## Split train and test
    n = int((len(y_test))/16)*16
    x_train = x_test[-n:]
    y_train = y_test[-n:]

    idx = range(len(y_train))
    test_idx = np.random.choice(idx, 64, replace=False)
    train_idx = [i for i in idx if i not in test_idx]

    x_test = x_train[test_idx]
    y_test = y_train[test_idx]
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    ## Evaluate un-trained model
    scoreUT = GDS.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
    scoreUT = dict(zip(GDS.metrics_names, scoreUT))
    scoreUT = pd.DataFrame([scoreUT])
    scoreUT.to_csv('db/results/best_score/GRU12_' + train + '_' + test + '_untrained_score.csv')
    #answerUT = GDS.predict(x_test, batch_size=batch_size)

    date = str(datetime.today().date())
    filepath = "db/results/GRU12_" + train + '_' + test + "_all_weights.h5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)
    filepath = "db/results/best_weights/GRU12_" + train + '_' +  test  + "_best_weights.h5"
    best_weights = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True, monitor="val_acc", mode='max')
    csv_logger = CSVLogger('db/results/logs/GRU12_' + train + '_' +  test + '_' + date + '_training_log.csv', separator=',', append=True)

    maxTime = 0
    i = 0
    # Training
    histories = []
    time1 = time.time()
    elasped = time1 - timeStart
    while (elasped+maxTime<600):
        time3 = time.time()
        i = i+1
        history = GDS.fit(x_train, y_train,
                            batch_size=batch_size,
                            verbose=1, epochs=1,
                            shuffle=False,  # turn off shuffle to ensure training data patterns remain sequential
                            callbacks=[checkpointer, csv_logger, best_weights],
                            validation_data=(x_test, y_test))
        histories.append(history.history)
        time2 = time.time()
        maxTime = max([maxTime, time2-time3])
        elasped = time2 - timeStart

    ## Output histories
    histories = pd.DataFrame(histories)
    histories.to_csv('db/results/history/GRU12_' + train + '_' + test + '_histories.csv')

    ## Evaluating on best results
    GDS.load_weights(filepath=filepath)
    score = GDS.evaluate(x_test, y_test, batch_size=16, verbose=1)
    score = dict(zip(GDS.metrics_names, score))
    score = pd.DataFrame([score])

    timeEnd = time.time()
    timeTotal = timeEnd-timeStart
    score['time'] = timeTotal
    score['counter'] = i
    score.to_csv('db/results/best_score/GRU12_' + train + '_' + test + '_best_score.csv')
    return scoreUT, score, timeTotal, i

## Load saved data
def load(filename):
    x_train = pkl.load(open('db/serialized/'+filename+'_x_train.np', 'rb'))
    x_test = pkl.load(open('db/serialized/'+filename+'_x_test.np', 'rb'))
    y_train = pkl.load(open('db/serialized/'+filename+'_y_train.np', 'rb'))
    y_test = pkl.load(open('db/serialized/'+filename+'_y_test.np', 'rb'))
    return (x_train, x_test, y_train, y_test)

## Model 2
def IKPP_model(train, test, rst):
    timeStart = time.time()
    ## Trim data
    x_test, y_test = load_test(test)

    ## Network structure
    nb_timesteps = 1
    nb_features = x_test.shape[1]
    output_dim = 1

    ## cross-validated model parameters
    batch_size = 16
    dropout = 0.25
    activation = 'sigmoid'
    nb_hidden = 128
    decoder_nb_hidden = 128
    initialization = 'glorot_normal'

    ##  reshape to appropriate dimensions
    x_test = np.resize(x_test, (x_test.shape[0], nb_timesteps, x_test.shape[1]))
    y_test = np.resize(y_test, (y_test.shape[0], output_dim))

    ## Split train and test
    n = int((len(y_test))/16)*16
    x_train = x_test[-n:]
    y_train = y_test[-n:]

    idx = range(len(y_train))
    test_idx = np.random.choice(idx, 64, replace=False)
    train_idx = [i for i in idx if i not in test_idx]

    x_test = x_train[test_idx]
    y_test = y_train[test_idx]
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]

    ## Load model (functional API)
    K.clear_session()
    GDS = load_model('db/results/model_' + train + '.h5')
    GDS.load_weights('db/results/best_weights/' + train + '_best_weights.h5')
    GDS.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    ## recreate model
    model = Sequential()

    model.add(Masking(mask_value=0., batch_input_shape=(batch_size, nb_timesteps, nb_features),
                      name='Mask'))  # embedding for variable input lengths

    ## Decoder
    model.add(GRU(decoder_nb_hidden, return_sequences=True, stateful=True, init=initialization, name="Encoder"))
    model.add(GRU(decoder_nb_hidden, return_sequences=True, stateful=True, init=initialization, name="Decoder"))
    model.add(Dense(GDS.layers[0].input_shape[2], activation="tanh", use_bias= False, name='Bridge'))

    ## GDS copy
    model.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU01',
                  batch_input_shape=(GDS.input_shape)))
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


    ## Preload weights
    for layer in GDS.layers[1:]:
        model.get_layer(layer.name).set_weights(layer.get_weights())

    ## Set un-trainable layers 4:-6
    for layer in model.layers[4:-6]:
        layer.trainable = False

    ## Reset weights (optional)
    reset = rst
    if reset == 1:
        for layer in model.layers[-6:]:
            layer.reset_states()

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    del GDS

    scoreUT = model.evaluate(x_test, y_test, verbose=1, batch_size=16)
    scoreUT = dict(zip(model.metrics_names, scoreUT))
    scoreUT = pd.DataFrame([scoreUT])
    scoreUT.to_csv('db/results/best_score/DE_GRU12_' + train + '_' + test + '_untrained_score.csv')

    # Prepare model checkpoints and callbacks
    date = str(datetime.today().date())
    filepath="db/results/DE_GRU12_" + train + '_' + test + str(rst) + "_all_weights.h5"
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=False)
    filepath = "db/results/best_weights/DE_GRU12_" + train + '_'  + test + str(rst) + '_' + "_best_weights.h5"
    best_weights = ModelCheckpoint(filepath=filepath, verbose=0, save_best_only=True, monitor="val_acc", mode='max')
    csv_logger = CSVLogger('db/results/logs/DE_GRU12_' + train + '_'  + test + str(rst) + '_' + date + '_training_log.csv', separator=',', append=True)

    maxTime = 0
    i = 0
    # Training
    histories = []
    time1 = time.time()
    elasped = time1 - timeStart
    while (elasped+maxTime<600):
        time3 = time.time()
        i = i+1
        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            verbose=1, epochs=1,
                            shuffle=False,  # turn off shuffle to ensure training data patterns remain sequential
                            callbacks=[checkpointer, csv_logger, best_weights],
                            validation_data=(x_test, y_test))
        histories.append(history.history)
        time2 = time.time()
        maxTime = max([maxTime, time2-time3])
        elasped = time2 - timeStart

    ## Output histories
    histories = pd.DataFrame(histories)
    histories.to_csv('db/results/history/DE_GRU12_' + train + '_' + test + str(rst) + '_histories.csv')

    ## Evaluating on best results
    model.load_weights(filepath=filepath)
    score = model.evaluate(x_test, y_test, batch_size=16, verbose=1)
    score = dict(zip(model.metrics_names, score))
    score = pd.DataFrame([score])

    timeEnd = time.time()
    timeTotal = timeEnd-timeStart
    score['time'] = timeTotal
    score['counter'] = i
    score.to_csv('db/results/best_score/DE_GRU12_' + train + '_' + test + str(rst) + '_best_score.csv')
    return scoreUT, score, timeTotal, i

## Main
references = ['train_dji', 'train_s&p', 'train_wilshere']
references2 = ['train_amazon', 'train_tesla', 'train_google', 'train_oil', 'train_gold']
filenames = ['amazon_amazon', 'tesla_tesla', 'google_google', 'oil_oil', 'gold_gold']

samplesUT, samples, timesT, counters = [], [], [], []


for reference in references:
    for filename in filenames:
        print(reference)
        print(filename)
        sampleUT, sample, timeT, counter = GRU12(reference, filename)
        samplesUT.append(sampleUT)
        samples.append(sample)
        timesT.append(timeT)
        counters.append(counter)

for i in range(len(filenames)):
    print(references2[i])
    print(filenames[i])
    sampleUT, sample, timeT, counter = GRU12(references2[i], filenames[i])
    samplesUT.append(sampleUT)
    samples.append(sample)
    timesT.append(timeT)
    counters.append(counter)

for reference in references:
    for filename in filenames:
        print(reference)
        print(filename)
        sampleUT, sample, timeT, counter = IKPP_model(reference, filename, 1)
        samplesUT.append(sampleUT)
        samples.append(sample)
        timesT.append(timeT)
        counters.append(counter)

for i in range(len(filenames)):
    print(references2[i])
    print(filenames[i])
    sampleUT, sample, timeT, counter = IKPP_model(references2[i], filenames[i], 1)
    samplesUT.append(sampleUT)
    samples.append(sample)
    timesT.append(timeT)
    counters.append(counter)

dbSamplesUT, dbSamples, dbTimeT, dbCounters = {}, {}, {}, {}
train, test = [], []

test = 'oil_oil'
train = "train_dji"

def plot_models(test):
    K.clear_session()

    ## GDS
    GDS = load_model('db/results/model_' + train + '.h5')
    GDS.compile(optimizer='rmsprop',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    GDS.name = "GDS"
    plot_model(GDS, 'db/models/GDS.png')

    x_train, x_test, y_train, y_test = load(test)

    ## Network Structure
    nb_timesteps = 1
    nb_features = x_train.shape[1]
    output_dim = 1

    ## cross-validated model parameters
    batch_size = 16
    dropout = 0.25
    activation = 'sigmoid'
    nb_hidden = 128
    decoder_nb_hidden = 1000
    initialization = 'glorot_normal'

    ## GDS Component
    comp = Sequential(name="GDS Component")
    comp.add(GRU(nb_hidden, return_sequences=True, stateful=True, init=initialization, name='GRU',
                  batch_input_shape=(GDS.input_shape)))
    comp.add(Dropout(dropout, name='DO'))
    plot_model(comp, 'db/models/GDS_component.png')


    ## Mask layer
    Mask = Sequential(name="Masking")
    Mask.add(Masking(mask_value=0., batch_input_shape=(batch_size, None, nb_features),
                      name='Mask'))  # embedding for variable input lengths
    plot_model(Mask, 'db/models/Masking.png')


    ## Decoder
    DE = Sequential(name='DecoderEncoder')
    DE.add(GRU(decoder_nb_hidden, return_sequences=True, stateful=True, init=initialization, name="Encoder"))
    DE.add(GRU(decoder_nb_hidden, return_sequences=True, stateful=True, init=initialization, name="Decoder"))
    DE.add(Dense(GDS.layers[0].input_shape[2], activation="linear", use_bias=False, name='Bridge'))
    plot_model(DE, 'db/models/DecoderEncoder.png')

    merged = Sequential(name='Merged')
    merged.add(Mask)
    merged.add(DE)
    merged.add(GDS)
    plot_model(merged, 'db/models/Merged.png')
