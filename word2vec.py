## Activate VirtualEnv
#activate_venv = 'C:/Users/Archie Wiranata/PycharmProjects/VirtualEnv/Scripts/'

## Imports ##
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors as kv
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import  nltk
import csv
import os
import  re

## Move to working directory
os.chdir('C:/Users/Archie Wiranata/PycharmProjects/Modelling')

## Google News load bin ##
def google_news_vocab():
    w2v = kv.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True) # load pre-trained word2vec by Google news
    vocab = list(w2v.vocab.keys()) # get vocabulary list from w2v
    return (set(vocab))
vocab_list = google_news_vocab()

## Google news dataset write vocab to text
with open('google_vocab.csv', 'w', newline='', encoding='utf-8') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(vocab_list)

## Google news 500000 words ##
w2v = kv.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, limit=500000)
simple_vocab_list = list(w2v.vocab.keys())
text_file = open('simple-google_vocab_text.txt', 'w', encoding='utf-8')
text_file.write()


## Open google vocab (optimal) ##
file = open('google_vocab_text.txt', 'r', encoding='utf-8-sig')
vocab_list = file.read().split(sep='","')
df_vocab = pd.DataFrame({'original':vocab_list})
df_vocab['processed'] = df_vocab['original'].str.replace(r'[^\w\s]+', '')

## Import data
def import_global_data(filename):
    df =  pd.DataFrame(pd.read_csv(filename))
    df.columns = ['index', 'date', 'label', 'topic']
    df.drop('index', axis=1, inplace=True)
    #df['topic'] = [str(item).lstrip() for item in df['topic']]
    #df['topic'] = [str(item).lstrip('b') for item in df['topic']]
    #df['topic'] = df['topic'].str.replace('"', '')
    #df['topic'] = df['topic'].str.replace("'", '')
    #df['topic'] = df['topic'].str.replace(r'[^\w\s]+', '')
    #df['topic'] = df['topic'].str.lower()
    return df

filename = 'db/train_dji.csv'
global_data = pd.read_csv(filename, encoding='ISO-8859-1')

nltk.download(['stopwords', 'wordnet'])
stopwords = nltk.corpus.stopwords.words('english')

feature_extraction = TfidfVectorizer(stop_words=stopwords, lowercase=True)
X_train = feature_extraction.fit_transform(global_data['topic'])
data_vocab = list(feature_extraction.vocabulary_.keys())
vocab_vectors = list(feature_extraction.vocabulary_.values())
df_data_vocab = pd.DataFrame({'word':data_vocab, 'vector':vocab_vectors})
mixed_vocab = pd.merge(df_vocab, df_data_vocab, left_on='processed', right_on='word')
mixed_vocab = mixed_vocab.sort_values(by=['vector'])
shared_rate = len(mixed_vocab)/len(df_data_vocab)


shared_words = set(data_vocab) & set(vocab_list_processed)
unshared_words = list((Counter(data_vocab) - Counter(shared_words)).elements())
print(len(shared_words)/ len(data_vocab))


def convert_sentences_to_unique_words(sentences): ## Build unique vocab list from datasets
    return (set(' '.join(sentences).split()))


def used_vocab(google_vocab, list_of_sentences):
    data_vocab = convert_sentences_to_unique_words([sentence for sentences in list_of_sentences for sentence in sentences])
    out = data_vocab & google_vocab
    print (len (out) / len(data_vocab))
    return (out)

total_vocab = used_vocab(vocab_list, [global_data['topic']]) ## Keyed vector from google dataset not reliable enough for a dictionary

def test():
    words = ['tesla', 'oil', 'world', 'finance', 'random', 'sad']
    vectors = []
    for word in words:
        vectors.append(w2v[word])
    print(vectors)

## Testing ##
def example():
    path = get_tmpfile("word2vec.model") # create temp model
    model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4) # train
    model.save("word2vec.model") # save temp model

    vector = model.wv['computer'] # Save a vector
    keyed_vectors = model.wv # save all Keyedvctors

### Main Script ###

#vocab_list = google_news_vocab()
