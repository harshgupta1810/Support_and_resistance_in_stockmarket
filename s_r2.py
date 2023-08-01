'''#%% md

1).Support describes levels where downward trends are halted
2).Resistance describes levels where upward trends are halted
3).Support levels can also be called ‘demand areas’
4).Resistance levels can be called ‘supply areas’
5).When support levels are broken they become new resistance levels;
6).When resistance levels are broken they become new support levels;
7).Market psychology is a factor and price trends “remember” previous support and resistance levels.

#%%
'''
import pandas as pd
import numpy as np
import yfinance
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import date
import matplotlib
from sklearn.cluster import KMeans
import plotly.graph_objects as go

matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)

def get_data(ticker,start):
    ticker = yfinance.Ticker(ticker)
    end = date.today()
    df = ticker.history(interval="1d",start=start,end=end)
    df = df.loc[:,['Open', 'High', 'Low', 'Close']]
    return df

def create_clusters(ticker,start):
    df = get_data(ticker,start)
    prices = np.array(df["Close"])
    # Perform cluster analysis
    K = 6
    kmeans = KMeans(n_clusters=6).fit(prices.reshape(-1, 1))
    # predict which cluster each price is in
    clusters = kmeans.predict(prices.reshape(-1, 1))
    return clusters,prices

def min_max_values(ticker,start):
    clusters,prices = create_clusters(ticker,start)
    # Create list to hold values, initialized with infinite values
    min_max_values = []
    # init for each cluster group
    for i in range(6):
        # Add values for which no price could be greater or less
        min_max_values.append([np.inf, -np.inf])
    # Print initial values
    print(min_max_values)
    # Get min/max for each cluster
    for i in range(len(prices)):
        # Get cluster assigned to price
        cluster = clusters[i]
        # Compare for min value
        if prices[i] < min_max_values[cluster][0]:
            min_max_values[cluster][0] = prices[i]
        # Compare for max value
        if prices[i] > min_max_values[cluster][1]:
            min_max_values[cluster][1] = prices[i]
    output = []
    # Sort based on cluster minimum
    s = sorted(min_max_values, key=lambda x: x[0])
    # For each cluster get average of
    for i, (_min, _max) in enumerate(s):
        # Append min from first cluster
        if i == 0:
            output.append(_min)
        # Append max from last cluster
        if i == len(min_max_values) - 1:
            output.append(_max)
        # Append average from cluster and adjacent for all others
        else:
            output.append(sum([_max, s[i + 1][0]]) / 2)
    return output

def plot_all(ticker, start):
    df = get_data(ticker,start)
    levels = min_max_values(ticker,start)

    # Create a plot of the candlestick data using mplfinance
    mpf.plot(df, type='candle', volume=False, style='charles', figratio=(10, 6), datetime_format='%d-%m-%Y',
             hlines=dict(hlines=levels, colors='b'))

    plt.show()



if __name__ == '__main__':
    a = min_max_values('LT.NS','2020-01-01')
    print(a)
    plot_all('LT.NS','2020-01-01')

