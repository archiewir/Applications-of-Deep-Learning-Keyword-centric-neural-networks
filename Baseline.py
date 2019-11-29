## set up imports
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

## Input the data
data = pd.read_csv('C:/Users/Archie Wiranata/PycharmProjects/Modelling/Combined_News_DJIA.csv')

## Clean the data
for i in range (2, data.shape[1], 1):
    data.iloc[:, i] = data.iloc[:, i].str.lower()
    v = data.iloc[:, i] .ravel()



## Split train and test
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

