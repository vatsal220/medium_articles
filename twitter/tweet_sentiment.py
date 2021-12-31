# necessary imports for the project
import tweepy as tp
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import string

from textblob import TextBlob
from datetime import datetime
from dotenv import load_dotenv

def env_reader(env_path):
    '''
    This function will read the environment variables given the location of where the
    .env file is located
    '''
    if not load_dotenv(env_path):
        return print('Environment File Not Found')
    return print('Loaded!')

def connect_twitter(api_key, secret_key, access_token, secret_access_token):
    '''
    This function will create a connection to the twitter api using the necessary
    credentials associated to the project.
    
    params:
        api_key (String) : Taken from twitter developer account
        secret_key (String) : Taken from twitter developer account
        access_token (String) : Taken from twitter developer account
        secret_access_token (String) : Taken from twitter developer account
         
    returns:
        This function will return an API which can be called
    '''
    auth = tp.OAuthHandler(
        consumer_key = api_key, 
        consumer_secret = secret_key
    )
    auth.set_access_token(
        key =  access_token, 
        secret = secret_access_token
    )
    api = tp.API(auth)
    
    try:
        api.verify_credentials()
        print("Connection to Twitter established.")
    except:
        print("Failed to connect to Twitter.")
    return api

def mk_dir(path, date):
    '''
    The purpose of this function is to make a directory if one does not exist of the current
    date in the data folder.
    
    params:
        path (String) : The path to the data folder
        date (String) : The current date yyyy-mm-dd
        
    returns:
        This function will do nothing if the folder already exists, otherwise it will create
        the folder
    
    example:
        mk_dir(path = './data/', date = '2021-09-23')
    '''
    
    # check if directory exists 
    exists = os.path.exists(path + date)
    
    if not exists:
        os.makedirs(path + date)
        print("New Directory Created")
        
def get_all_tweets(screen_name, api, today):
    '''
    This function will get the latest ~3200 tweets associated to a twitter screen name.
    It will proceed to get the tweet id, created at and the content and store it in a df.
    It will save the associated results in a CSV file.
    
    params:
        screen_name (String) : The twitter handle associated to the user you want to get
                               tweets from
        api (API) : The tweepy API connection
        today (String) : Todays date in string format
    
    returns:
        This function will return a df associated to the tweet id, created_at and content
        
    source:
        [yanofsky](https://gist.github.com/yanofsky/5436496)
    '''
    #initialize a list to hold all the tweepy Tweets
    alltweets = []  
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
    
    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
#         print(f"getting tweets before {oldest}")
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        
#         print(f"...{len(alltweets)} tweets downloaded so far")
    
    #transform the tweepy tweets into a 2D array that will populate the csv 
    outtweets = [
        [
            t.author.name, t.id_str, t.created_at, t.text, t.entities.get('hashtags'), t.author.location,
            t.author.created_at, t.author.url, t.author.screen_name, t.favorite_count, t.favorited,
            t.retweet_count, t.retweeted, t.author.followers_count, t.author.friends_count
        ] for t in alltweets
    ]
    
    #write the csv  
    cols = [
        'author_name', 'tweet_id', 'tweet_created_at', 'content', 'hashtags', 'location',
        'author_created_at', 'author_url', 'author_screen_name', 'tweet_favourite_count', 'tweet_favourited', 
        'retweet_count', 'retweeted', 'author_followers_count', 'author_friends_count'
    ]
    df = pd.DataFrame(outtweets, columns = cols)
    mk_dir(path = './data/', date = today)
    df.to_csv('./data/{}/{}_tweets_{}.csv'.format(today, screen_name, today), index = False)
    time.sleep(10)
    return df

def read_tweet_data(path):
    '''
    This function will identify if todays data has already been scraped from the twitter API.
        - If it has been scraped, this function will read all the scraped data from today 
           and previous days
    Upon fetching all the data, it will drop duplicates on the tweet_id and tweet_created_at
    columns to remove duplicated tweets scraped from previous days.
    
    params:
        path (String) : The path to the data folder
        today (String) : Today's date in string format yyyy-mm-dd
        
    returns:
        This function will return the tweets_df associated to tweets from all handles over 
        the past few months
        
    example:
        read_tweet_data(
            path = './data/'
        )
    '''
    
    # get all non hidden subdirectories from the path
    sub_dir = [d for d in os.listdir(path) if d[0] != '.']
    
    # read csv from all sub directories and concat results
    files = []
    for d in sub_dir:
        for p in os.listdir(path + d):
            if p[0] != '.':
                files.append(path + d + '/' + p)

    read_csvs = []
    for file in files:
        read_csvs.append(pd.read_csv(file, converters={'hashtags': eval}, encoding='utf-8-sig'))

    read_csvs = pd.concat(read_csvs)
    tweets_df = read_csvs.drop_duplicates(subset = ['tweet_id', 'tweet_created_at'])
    return tweets_df

def remove_punctuation(tweet):
    '''
    This function will remove all punctuations from the tweet passed in
    '''
    return ''.join(ch for ch in tweet if ch not in set(string.punctuation))

def remove_sw(tweet):
    '''
    This function will remove all stopwords from the tweet passed in
    '''
    sw = [
        'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of',
 'most', 'itself', 'other', 'off', 'is', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until',
 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this',
 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at',
 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because',
 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too',
 'only', 'myself', 'which', 'those', 'i', 'after', 'few',
 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'
    ]
    tweet=tweet.lower()
    tweet = ' '.join([w for w in tweet.split(' ') if w not in sw])
    return tweet

def tweet_sentiment(tweet):
    '''
    Identify the sentiment associated to a tweet.
    
    params:
        tweet (String) : The stirng you want the sentiment of
        
    returns:
        A score between -1 and 1, where values greater than 0
        would indicate a positive sentiment and values less
        than 0 would be negative. Values = 0 is a neutral
        sentiment tweet.
    '''
    tb = TextBlob(tweet)
    score = tb.sentiment.polarity
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    # constants
    PATH = os.path.expanduser('~') + '/'
    env_file = '.env'
    today = datetime.today().strftime('%Y-%m-%d')
    
    # read environment file
    env_reader(os.getcwd() + '/' + env_file)

    # connect to twitter API
    api = connect_twitter(
        api_key = os.getenv("twitter_api_key"), 
        secret_key = os.getenv("twitter_secret"),
        access_token = os.getenv("twitter_access_token"),
        secret_access_token = os.getenv("twitter_access_token_secret")
    )

    # handles we're scraping
    handles = [
        'TorontoPolice', 'HamiltonPolice', 'YRP', 'PeelPolice'
    ]

    if today not in os.listdir('./data/'):
        for user in handles:
            print(user)
            _ = get_all_tweets(user, api, today)

    tweets_df = read_tweet_data(
        path = './data/'
    )
    print(tweets_df.shape)
    
    # clean tweets
    tweets_df['cleaned_tweet'] = tweets_df['content'].apply(remove_punctuation)
    tweets_df['cleaned_tweet'] = tweets_df['content'].apply(remove_sw)
    
    # get sentiments
    tweets_df['tweet_sentiment'] = tweets_df['cleaned_tweet'].apply(tweet_sentiment)

    plt.clf()
    tweets_df['tweet_sentiment'].value_counts().plot(kind = 'barh')
    plt.title('Sentiment of Tweets')
    plt.xlabel('Frequency of Tweet Sentiment')
    plt.show()
    
if __name__ == '__main__':
    main()