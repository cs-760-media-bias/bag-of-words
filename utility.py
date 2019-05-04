import json
import os
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer
import nltk
print('Downloading natural language toolkit packages as needed...\n')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def importTidy(path_to_tweets, path_to_bias):
    '''
    df = importTidy(path_to_tweets, path_to_bias)
    '''
    with open(path_to_bias) as f:
        bias = json.load(f)

    json_files = [pos_json for pos_json in os.listdir(path_to_tweets) if pos_json.endswith('.json')]

    df=pd.DataFrame()
    for file in json_files:
        with open('data/tweets_tidy/'+file, 'r') as f:
            datastore = json.load(f)
            df_new = pd.DataFrame(datastore["tweets"])
            #df_new["tweetSource"]=file.rsplit(".")[0]
            df_new["tweetSource"]=datastore["user"]["screen_name"]
            df_new["followers_count"]=datastore["user"]["followers_count"]
            df_new["friends_count"]=datastore["user"]["friends_count"]
            df_new["listed_count"]=datastore["user"]["listed_count"]
            df_new["statuses_count"]=datastore["user"]["statuses_count"]

            for source in bias["sources"]:
                if file.rsplit(".")[0] in source["twitter_handles"]:
                    df_new["ad_fontes_x"] = source["ad_fontes_x"]
                    df_new["ad_fontes_y"] = source["ad_fontes_y"]

            df=df.append(df_new, ignore_index=True)

    #split up datetime
    df = datetime(df)
    #ensure numeric columns are numeric and not string
    df = numericColumns(df)
    #map month and weekday to numeric representation
    df = mapDate(df)

    #move ad_fontes_x/y to front
    df = df[['ad_fontes_x', 'ad_fontes_y'] +
       [c for c in df if c not in ['ad_fontes_x', 'ad_fontes_y']]]

    #get count of hashtags
    df["hashtag_count"]=df.apply(lambda row: len(row.hashtags), axis=1)

    return df


def datetime(df):
    '''
    df = datetime(df)

    Takes in a pandas dataframe of tweets and splits the created_at column into
    five columns, "Weekday", "Month", "Day", "Time", and "Year"

    input:
        - df: a pandas dataframe of tweets

    output:
        - df: a pandas dataframe of tweets
    '''

    #split datetime column into multiple columns
    datetime = df["created_at"].str.split(" ", n = 6, expand = True)
    datetime.columns = ["tweetWeekday", "tweetMonth", "tweetDayOfMonth", "tweetTime","useless","tweetYear"]

    #drop the column of +0000
    datetime = datetime.drop(["useless"], axis=1)

    #drop the old "created_at" column and add the new columns in
    df = df.drop(["created_at"], axis=1)
    df = pd.concat([df, datetime], axis=1, sort=False)

    #move order of columns
    cols_in_front = ["id", "tweetWeekday", "tweetMonth", "tweetDayOfMonth", "tweetTime", "tweetYear"]
    df = df[[c for c in cols_in_front if c in df]
       + [c for c in df if c not in cols_in_front]]

    return df


def mapDate(df):
    '''
    df = mapDate(df)

    Maps the month and day-of-week columns to their numeric equivalent. Sunday=1

    Columns that are converted:
        - tweetWeekday
        - tweetMonth

    input:
        - df: a pandas dataframe containing the columns to be converted

    output:
        - df: the same pandas dataframe after conversion
    '''

    mo = {'JAN':1, 'FEB':2, 'MAR':3, 'APR':4, 'MAY':5, 'JUN':6, 'JUL':7,
          'AUG':8, 'SEP':9, 'OCT':10, 'NOV':11, 'DEC':12}
    wd = {'SUN':1, 'MON':2, 'TUE':3, 'WED':4, 'THU':5, 'FRI':6, 'SAT':7}

    df.tweetMonth = df.tweetMonth.str.upper().map(mo)
    df.tweetWeekday = df.tweetWeekday.str.upper().map(wd)

    return df


def numericColumns(df):
    '''
    df = numericColumns(df)

    Takes columns that are originally imported as strings and converting them
    to numeric features.

    Columns that are converted:
        - tweetDayOfMonth
        - tweetYear
        - id
        - photo_count
        - video_count

    input:
        - df: a pandas dataframe containing the columns to be converted

    output:
        - df: the same pandas dataframe after conversion

    '''

    #transfer day of month and year to numeric
    df['tweetDayOfMonth'] = pd.to_numeric(df['tweetDayOfMonth'])
    df['tweetYear'] = pd.to_numeric(df['tweetYear'])
    df['id'] = pd.to_numeric(df['id'])
    df['photo_count'] = pd.to_numeric(df['photo_count'])
    df['video_count'] = pd.to_numeric(df['video_count'])

    return df


def cleanTweet(messy_tweet):
    '''
    clean_tweet = cleanTweet(messy_tweet)

    Reads a string representing a single tweet and performs the following:
        0. Remove URLs
        1. Split tweet into tokens
        2. Converts to lowercase
        3. Removes punctuation
        4. Removes newline characters
        5. Filters out remaining non-alphabetic characters
        6. Filters out stop words
        7. Lemmatizes tokens

    inputs:
        messy_tweet: a string

    outputs:
        clean_tweet: a string
    '''

    #lemmitizer
    wordnet_lemmatizer = WordNetLemmatizer()

    text = re.sub(r'http\S+', '', messy_tweet, flags=re.MULTILINE)

    tokens = nltk.word_tokenize(text)

    #lower case
    tokens = [x.lower() for x in tokens]

    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    stripped = [x.translate(table) for x in tokens]

    # remove non-alphabetic
    words = [word for word in stripped if word.isalpha()]

    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [x for x in words if not x in stop_words]

    #lemmatize words
    lemmed = [wordnet_lemmatizer.lemmatize(word) for word in words]

    #return full tweet
    return ' '.join(lemmed)
