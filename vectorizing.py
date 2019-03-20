## Imports
from newsapi import NewsApiClient
import numpy as np
import pandas as pd
import math
import os
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

newsapi = NewsApiClient(api_key= "aa5becdd584a4cfd8c72af71fe5571fb")
os.chdir('C:/Users/Archie Wiranata/PycharmProjects/Modelling')

def sourceList():
    sources = pd.DataFrame(newsapi.get_sources())
    sources = sources["sources"]
    index = ['category', 'country', 'description', 'id', 'language', 'name', 'url']
    category = []
    country = []
    description = []
    id = []
    language = []
    name = []
    url = []

    for source in sources:
        category.append(source['category'])
        country.append(source['country'])
        description.append(source['description'])
        id.append(source['id'])
        language.append(source['language'])
        name.append(source['name'])
        url.append(source['url'])
    df = pd.DataFrame([category, country, description, id, language, name, url], index=index)
    df = df.T


    return (df)

sources = sourceList() # in dataframe
#sources.to_csv('sources.csv', sep=',', encoding='utf-8', index=False)
## remove # to make new csv file

lan = [sources.language == 'en']
cat = [sources.category == 'business']

select = np.logical_and(cat, lan).flatten()
business_en = sources.iloc[select,]
business_en = business_en[business_en.id != 'business-insider-uk']
id_url = dict(zip(business_en['id'], business_en['url']))
id = ', '.join(business_en['id'])
url = ', '.join(business_en['url'])

## Filter out domains and sources

example = {'keyword':'bitcoin', 'source':'bbc-news, the-verge', 'domain':'bbc.co.uk, techcrunch.com'}

def parameters():
    ## Keyword ##
    keyword = input("What keyword would you like to use for your news headlines?")

    ## Time ##
    date_range = [(datetime.today() - relativedelta(months=+6)), datetime.today()]
    time = input("Preset or Custom input for datetime of test data? (P/C)")

    # Custom
    if time == "C":
        print("Within what datetime would you like to use as test data? (Within the ast 6 months)")
        while True:
            ## year ##
            while True:
                try:
                    year = int(input("Year (eg. 2019): "))
                except ValueError:
                    print ("That's not a number!")
                else:
                    if date_range[0].year <= year <= date_range[1].year:
                        print(str(year) + "-MM-DD hh:mm")
                        break
                    else:
                        print("Out of range, try again. Range: ")
                        print(str(date_range[0].year) + " - " + str(date_range[1].year))

            ## mm ##
            while True:
                try:
                    mm = int(input("Month (eg. 1, 12): "))
                except ValueError:
                    print (" That is not a number!")
                else:
                    if 1 <= mm <= 12:
                        print(str(year) + "-" + str(mm) + "-DD hh:mm")
                        break
                    else:
                        print ("Invalid month.")

            ## dd ##
            while True:
                try:
                    dd = int(input("date (eg. 1, 31): "))
                except ValueError:
                    print("That is not a number!")
                else:
                    if 1<= dd <= 31:
                        print(str(year) + "-" + str(mm) + "-" + str(dd)+ " hh:mm")
                        break
                    else:
                        print("Invalid date, try again.")

            ## hour ##
            while True:
                try:
                    hour = int(input("hour (eg. 0, 23): "))
                except ValueError:
                    print("That is not a number!")
                else:
                    if 0<= hour <= 23:
                        print(str(year) + "-" + str(mm) + "-" + str(dd)+ " " + str(hour) + ":mm")
                        break
                    else:
                        print("Invalid hour, try again")

            ## min ##
            while True:
                try:
                    minutes = int(input("minutes (eg. 0, 59): "))
                except ValueError:
                    print("That is not a number!")
                else:
                    if 0<= minutes <= 59:
                        print(str(year) + "-" + str(mm) + "-" + str(dd)+ " " + str(hour) + ":" + str(min))
                        break
                    else:
                        print("Invalid minute, try again.")

            try:
                time = datetime(year=year, month=mm, day=dd, hour=hour, minute=minutes)
            except ValueError:
                print("Date is invalid, try again")
            else:
                if date_range[0] <= time <= date_range[1]:
                    print(" Datetime chosen:")
                    print(str(time))
                    break
                else:
                    print("Date chosen is not within 6 month period from today.")
                    print(str(time))

    # Preset
    else:
        time = int(input("How many hours ago would you like your test data from? eg(1 , 24)"))
        time = datetime.today() - relativedelta(hours=time)
        print("Datetime chosen:")
        print((str(time)))

    ## Index ##
    indexes = ["DJI", "TSLA", "GOOGL", "AMZN"] ## Add 3 commodity indexes
    print ("Select one of the following index:")
    index = input(indexes)

    parameter = [keyword, time, index]
    return parameter

def convert_json (Json):
    column = ['author', 'source_name', 'source_id', 'url', 'publishedAt', 'content', 'description', 'urlToImage',
              'title']
    df = pd.DataFrame(columns=column)
    for article in Json['articles']:
        m = []
        m.append(article['author'])
        m.append(article['source']['name'])
        m.append(article['source']['id'])
        m.append(article['url'])
        m.append(article['publishedAt'])
        m.append(article['content'])
        m.append(article['description'])
        m.append(article['urlToImage'])
        m.append(article['title'])
        df.loc[df.shape[0] + 1] = m
    return df

def get_articles(keyword, source, domain, date1, date2):
    articles = newsapi.get_everything(q=keyword,
                                      sources=source,
                                      domains=domain,
                                      from_param=date1,
                                      to=date2,
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=100,
                                      page=1    )
    pages = int(math.ceil(articles['totalResults']/100))
    df_out = convert_json(articles)

    print(pages)
    if pages > 10: pages = 10
    for p in range(2,pages+1):
        articles = newsapi.get_everything(q=keyword,
                                          sources=source,
                                          domains=domain,
                                          from_param=date1,
                                          to=date2,
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=100,
                                          page=p)
        print (p)
        df = convert_json(articles)
        df_out = pd.concat([df_out,df], ignore_index=True)

    df_out['publishedAt'] = [t.replace('T', ' ') for t in df_out['publishedAt']]
    df_out['publishedAt'] = [z.replace('Z', '') for z in df_out['publishedAt']]
    return df_out

def per_source (keyword, sources):
    df_out = []
    for k, v in sources.items():
        df = get_articles(keyword, k, v)
        df_out.append(df)
    df_out = pd.DataFrame(pd.concat(df_out, axis=0))
    name = keyword + ' ' + str(datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + '.csv'
    df_out.to_csv(name, index=False)
    return df_out

def per_page(keyword, source, domain, date1, date2):

    articles = newsapi.get_everything(q=keyword,
                                      sources=source,
                                      domains=domain,
                                      from_param=str(date1),
                                      to=str(date2),
                                      language='en',
                                      sort_by='relevancy',
                                      page_size=100,
                                      page=1)
    pages = int(math.ceil(articles['totalResults'] / 100))
    df_out = convert_json(articles)
    print(pages)
    if pages > 10: pages = 10
    for p in range(2, pages + 1):
        articles = newsapi.get_everything(q=keyword,
                                          sources=source,
                                          domains=domain,
                                          from_param=str(date1),
                                          to=str(date2),
                                          language='en',
                                          sort_by='relevancy',
                                          page_size=100,
                                          page=p)
        df = convert_json(articles)
        df_out = pd.concat([df_out, df], ignore_index=True)

    df_out['publishedAt'] = [t.replace('T', ' ') for t in df_out['publishedAt']]
    df_out['publishedAt'] = [z.replace('Z', '') for z in df_out['publishedAt']]
    return pd.DataFrame(df_out)

def per_date (keyword, source, domain):
    date2 = datetime(year=2019, month=2, day=11, hour=22)# -month=+1, (+days=+7, hours=+12), +(days=+15), +(days=+22, hours=+12)
    date3 = date2 + relativedelta(hours=+8)
    df_out = []

    # 8-hour data scrapping loop
    for i in range(0,13): # Max 21 8-hours period
        df = per_page(keyword, source, domain, date2, date3)
        date3 = date3 + relativedelta(hours=+8)
        date2 = date2 + relativedelta(hours=+8)
        df_out.append(df)

    df_out = pd.DataFrame(pd.concat(df_out))
    print (str(date2))
    df_out.to_csv('Master_5.csv', index=False)
    return (df_out)

bitcoin = get_articles(example['keyword'], example['source'], example['domain'])
bitcoin.to_csv('bit.csv', index=False)

time1 = datetime.now()
business = per_date('', id, url)

#per_source('', id_url)
time2 = datetime.now()
time_taken = time2 - time1

date1 = str(datetime(2019,1,15))
date2 = str(datetime(2019,2,1))
date3 = str(datetime(2019,2,15))

## mid Jan - Feb
tesla = get_articles('tesla', id, url, date1, date2)
tesla.to_csv('tesla1.csv', index=False)
elon_musk = get_articles('elon_musk', id, url, date1, date2)
elon_musk.to_csv('elon_musk1.csv', index=False)
google = get_articles('google', id, url, date1,date2)
google.to_csv('google1.csv', index=False)
amazon = get_articles('amazon', id, url, date1, date2)
amazon.to_csv('amazon1.csv', index=False)
jeff_bezos = get_articles('jeff_bezos', id, url, date1, date2)
jeff_bezos.to_csv('jeff_bezos1.csv', index=False)
oil = get_articles('oil', id, url, date1, date2)
oil.to_csv('oil1.csv', index=False)
gold = get_articles('gold', id, url, date1, date2)
gold.to_csv('gold1.csv', index=False)

## Fed - mid Feb
tesla = get_articles('tesla', id, url, date2, date3)
tesla.to_csv('tesla2.csv', index=False)
elon_musk = get_articles('elon_musk', id, url, date2, date3)
elon_musk.to_csv('elon_musk2.csv', index=False)
google = get_articles('google', id, url, date2,date3)
google.to_csv('google2.csv', index=False)
amazon = get_articles('amazon', id, url, date2, date3)
amazon.to_csv('amazon2.csv', index=False)
jeff_bezos = get_articles('jeff_bezos', id, url, date2, date3)
jeff_bezos.to_csv('jeff_bezos2.csv', index=False)
oil = get_articles('oil', id, url, date2, date3)
oil.to_csv('oil2.csv', index=False)
gold = get_articles('gold', id, url, date2, date3)
gold.to_csv('gold2.csv', index=False)

#rate = time_taken/ business.shape[0]

'australian-financial-review, bloomberg, business-insider, business-insider-uk, cnbc, financial-post, financial-times, fortune, the-economist, the-wall-street-journal'
