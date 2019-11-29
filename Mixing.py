## Imports ##
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import pickle as pkl
import os

## Move to working directory
os.chdir('C:/Users/Archie Wiranata/PycharmProjects/Modelling')

## Load data
master = pd.read_csv('db/Master_.csv', encoding = 'ISO-8859-1')
master['topic'] = master['title']
master = master[['Date', 'topic']]

stocknews = pd.read_csv('db/processed_stocknews.csv')
stocknews = stocknews[['Date', 'topic']]
df = pd.concat((stocknews, master), ignore_index=True, sort='Date')

## Load index
gold = pd.read_csv('db/Index/processed_gold.csv', encoding = 'ISO-8859-1')
amazon = pd.read_csv('db/Index/processed_amazon.csv', encoding = 'ISO-8859-1')
google = pd.read_csv('db/Index/processed_google.csv', encoding = 'ISO-8859-1')
oil = pd.read_csv('db/Index/processed_oil.csv', encoding = 'ISO-8859-1')
tesla = pd.read_csv('db/Index/processed_tesla.csv', encoding = 'ISO-8859-1')

db = {'gold': gold, 'amazon': amazon, 'google': google, 'oil': oil, 'tesla': tesla}

## Mix data
def mix(train, index):
    df = pd.merge(train, db[index], how='left', on='Date').dropna()
    df = pd.DataFrame(df[['Date', 'topic', 'Label']])
    df.to_csv('db/train_' + index + '.csv', index=False, index_label=False)

mix(df, 'gold')
mix(df, 'amazon')
mix(df, 'google')
mix(df, 'oil')
mix(df, 'tesla')