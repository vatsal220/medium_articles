import yfinance as yf 
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def fetch_stock_data(ticker_dct):
    '''
    This function will fetch stock data through the yahoo finance
    stock api.
    The ticker_dct will be of the following format : 
        - Key -> Ticker 
        - Values -> Start date, end date of the stock data
    
    params:
        ticker_dct (Dictionary) : The stocks you want to fetch the data for
        
    returns:
        This funciton will return a dictionary, the key will be the ticker
        and the value will be the data associated to that ticker over the
        specified user time period
    '''
    
    for k,v in ticker_dct.items():
        ticker_dct[k] = yf.download(k, v[0], v[1])
    return ticker_dct

def visualize_close_prices(dfs):
    '''
    This function will visualize the closing prices associated
    to a given input dataframe
    '''
    for k,v in dfs.items():
        plt.title(k)
        v.Close.plot()
        plt.show()

def main():
    # constants
    today = '2022-01-01'
    ticker_dct = {
        'SHOP' : ['2015-05-22',today],
        'COIN' : ['2021-04-16',today],
        'DOCN' : ['2021-03-26',today],
        'BMBL' : ['2021-02-12',today],
        'COUR' : ['2021-04-01',today],
        'DUOL' : ['2021-07-30',today]
    }
    
    dfs = fetch_stock_data(ticker_dct)
    
    # visualize trend lines
    visualize_close_prices(dfs)
    
    # find similar stocks
    # benchmark the dates to start from 0
    for ticker, df in dfs.items():
        dates = df.index.values
        date_map = {date:idx for idx, date in enumerate(dates)}

        dfs[ticker]['benchmark_date'] = dfs[ticker].index.map(date_map)
        
    shop_df = dfs['SHOP']
    distances = {}
    for k,v in dfs.items():
        if k != 'SHOP':
            max_d = v['benchmark_date'].max()
            benchmark = np.array(shop_df[shop_df['benchmark_date'].between(0, max_d)]['Close'].values)

            y = np.array(v.Close.values)
            d,p = fastdtw(benchmark, y)
            distances[k] = d
    
    return distances

if __name__ == '__main__':
    res = main()
    print(res)