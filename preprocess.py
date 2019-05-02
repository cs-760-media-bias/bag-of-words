import json
import sys
import os
import pandas as pd
import numpy as np
from utility import importTidy, datetime, cleanTweet, numericColumns
from bow import bagOfWords
from sklearn.model_selection import train_test_split
sys.path.append('data/tweets_tidy/')


print('Importing data...')
df = importTidy('data/tweets_tidy/', 'data/sources.json')

simplified = df.sample(frac=.25)
simplified = simplified.reset_index()

print('Cleaning tweets...')
text = simplified["text"].values
tweets_cleaned = []
for messy_tweet in text:
    clean_tweet = cleanTweet(messy_tweet)
    tweets_cleaned.append(clean_tweet)

print('Developing Bag-of-Words model...')
[bag_of_words, feature_names] = bagOfWords(tweets_cleaned)

# Create data frame
bow_df = pd.DataFrame(bag_of_words.toarray(), columns=feature_names)
bow_ed = pd.concat([simplified, bow_df], axis=1, sort=False)

y= bow_ed["source"]
X = bow_ed.drop(["id", "source", "reply_to_screen_name", "reply_to_tweet_id", "reply_to_user_id", "text", "urls", "user_mentions", "hashtags", "tweetTime", "index"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('Writing data...')
X_train.to_csv('preprocessed/X_train.csv', sep=',')
X_test.to_csv('preprocessed/X_test.csv', sep=',')
y_train.to_csv('preprocessed/y_train.csv', sep=',')
y_test.to_csv('preprocessed/y_test.csv', sep=',')
