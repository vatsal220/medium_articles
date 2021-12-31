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

# constants
PATH = os.path.expanduser('~') + '/'
env_file = '.env'
today = datetime.today().strftime('%Y-%m-%d')

# read environment file
def env_reader(env_path):
    '''
    This function will read the environment variables given the location of where the
    .env file is located
    '''
    if not load_dotenv(env_path):
        return print('Environment File Not Found')
    return print('Loaded!')

env_reader(os.getcwd() + '/' + env_file)

# connect to twitter API
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

api = connect_twitter(
    api_key = os.getenv("twitter_api_key"), 
    secret_key = os.getenv("twitter_secret"),
    access_token = os.getenv("twitter_access_token"),
    secret_access_token = os.getenv("twitter_access_token_secret")
)