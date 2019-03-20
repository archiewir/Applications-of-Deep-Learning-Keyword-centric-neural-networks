## import ##
import numpy as np
from datetime import datetime
import pandas as pd
import os

# Change to working directory
os.chdir('C:/Users/Archie Wiranata/PycharmProjects/Modelling')

## stocknews data ##
data = pd.read_csv('db\Combined_News_DJIA.csv')
stocknews = pd.DataFrame(columns=['Date', 'Label', 'topic'])
temp = data.iloc[:,0:2]

for i in range (2, data.shape[1]):
    right = data.iloc[:,i]
    temp['topic'] = right
    stocknews = stocknews.append(temp)

def clean_stocknews():
    df = stocknews
    df['topic'] = [str(item).lstrip() for item in df['topic']]
    df['topic'] = [str(item).lstrip('b') for item in df['topic']]
    df['topic'] = df['topic'].str.replace('"', '')
    df['topic'] = df['topic'].str.replace("'", '')
    df['topic'] = df['topic'].str.replace(r'[^\w\s]+', '')
    df['topic'] = df['topic'].str.lower()
    df.to_csv('db\processed_stocknews.csv', sep=',', index=False)
    return df


## Index data ##
def index_data(key):
    df = pd.read_csv(indexes[key])
    df = df[['Date', 'Adj Close']]
    df['Label'] = [1 if x>=0 else 0 for x in df['Adj Close'].diff()]
    name = 'db/processed_' + key + '.csv'
    df.to_csv(name, index=False)
    return df

indexes = {'dji': 'db\^DJI.csv','s&p':'db\^GSPC.csv', 'wilshere':'db\WILL5000INDFC.csv','amazon': 'db\AMZN.csv', 'google': 'db\GOOGL.csv', 'tesla': 'db\TSLA.csv', 'oil':'db\WTI.csv', 'gold':'db\XAU_USD Historical Data.csv'}

dji = index_data('dji')
sp500 = index_data('s&p')
amazon = index_data('amazon')
googl = index_data('google')
tesla = index_data('tesla')
oil = index_data('oil')

gold = pd.read_csv(indexes['gold'])
gold = gold[['Date', 'Price']]
gold['Label'] = [1 if x>=0 else 0 for x in pd.to_numeric(gold['Price'], errors='coerce').diff()]
gold['Date'] = [str(datetime.strptime(date, '%b %d, %Y').date()) for date in gold['Date']]
gold.to_csv('db/processed_gold.csv', index=False)

wilshere = pd.read_csv(indexes['wilshere'])
wilshere['Date'] = wilshere['DATE']
wilshere['Label'] = [1 if x>=0 else 0 for x in pd.to_numeric(wilshere['WILL5000INDFC'], errors='coerce').diff()]
wilshere.to_csv('db/processed_wilshere,.csv', index=False)

## News API data
def newsapi_data (name, files):
    df_out = []
    for i in range(1, files+1):
        filename = 'db/' +name + str(i) + '.csv'
        df = pd.read_csv(filename, encoding='"ISO-8859-1"')
        df_out.append(df)
    df_out = pd.DataFrame(pd.concat(df_out, ignore_index=True))
    df_out = df_out[['publishedAt', 'title']].dropna()
    df_out['Date'] = [str(datetime.strptime(str(dt), "%Y-%m-%d %H:%M:%S").date()) for dt in df_out['publishedAt']]
    filename = 'db/' + name + '.csv'
    df_out.to_csv(filename, index=False)
    return df_out

Master = newsapi_data('Master_', 5)

Elon_musk = newsapi_data('elon_musk', 2)
Tesla = newsapi_data('tesla', 2)
Amazon = newsapi_data('amazon', 2)
Jeff_bezos = newsapi_data('jeff_bezos', 2)
Google = newsapi_data('google', 2)

Oil = newsapi_data('oil', 2)
Gold = newsapi_data('gold', 2)

datasets = {'master': Master, 'elon_musk': Elon_musk, 'tesla': Tesla, 'amazon': Amazon, 'jeff_bezos': Jeff_bezos, 'google': Google, 'oil': Oil, 'gold': Gold}
indexes = {'dji': dji,'s&p':sp500, 'wilshere':wilshere,'amazon': amazon, 'google': googl, 'tesla': tesla, 'oil':oil, 'gold':gold}

def merge_index_newsapi(news_key, index_key):
    df = pd.DataFrame(pd.merge(datasets[news_key], indexes[index_key], how='left', on='Date', sort=True))
    df ['topic'] = df['title']
    df['test'] = df[['Date', 'title']].apply(lambda x: ' '.join(x), axis=1 )
    df = df.drop_duplicates(subset=['test'], keep='last')
    df = df[['Date', 'Label', 'topic']].dropna()
    name = 'db/' + news_key + '_' + index_key + '.csv'
    df.to_csv(name, index= False)
    return df

elon_musk_out = merge_index_newsapi('elon_musk', 'tesla')
tesla_out = merge_index_newsapi('tesla', 'tesla')
amazon_out = merge_index_newsapi('amazon', 'amazon')
jeff_bezos_out = merge_index_newsapi('jeff_bezos', 'amazon')
google_out = merge_index_newsapi('google', 'google')

oil_out = merge_index_newsapi('oil', 'oil')
gold_out = merge_index_newsapi('gold', 'gold')

def merge_pretrain_data(stocknews, newsapi):
    stocknews['title'] = stocknews['topic']
    stocknews = stocknews[['Date', 'title']].dropna()
    newsapi  = newsapi[['Date', 'title']].dropna()
    newsapi = newsapi.loc[newsapi['Date']>'2019-01-14']
    newsapi = newsapi.loc[newsapi['Date']<'2019-02-16']
    df = pd.DataFrame(pd.concat([newsapi, stocknews], ignore_index=True))

    return df

training_data = merge_pretrain_data(stocknews, Master)
datasets['train'] = training_data
train_dji = merge_index_newsapi('train', 'dji')
train_sp = merge_index_newsapi('train', 's&p')
train_wilshere = merge_index_newsapi('train', 'wilshere')