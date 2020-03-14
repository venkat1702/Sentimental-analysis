import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Twitter API Authentication Variables
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
tweets = api.search('topic which you want to search', count=200)
data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
print(data.head(10))
print(tweets[0].created_at)
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
listy = []
for index, row in data.iterrows():
  ss = sid.polarity_scores(row["Tweets"])
  listy.append(ss)
#storing the scores into a csv file using pandas
se = pd.Series(listy)
data['polarity'] = se.values
print(data.head(100))
