import pandas as pd
import numpy as np
import yfinance
from mplfinance.original_flavor import candlestick_ohlc
import mplfinance as mpf
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from datetime import date
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)

def get_stock_data(symbol, start_date):
    ticker = yfinance.Ticker(symbol)
    end_date = date.today()
    df = ticker.history(interval="1d", start=start_date, end=end_date)
    df['Date'] = pd.to_datetime(df.index)
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    df.index = pd.to_datetime(df.index, errors='coerce')

    df = df.loc[:,['Date', 'Open', 'High', 'Low', 'Close']]
    return df

def plot_support_resistance_levels(symbol, start_date):
    df = get_stock_data(symbol, start_date)
    def isSupport(df, i):
        support = df['Low'][i] < df['Low'][i-1] and df['Low'][i] < df['Low'][i+1] and df['Low'][i+1] < df['Low'][i+2] and df['Low'][i-1] < df['Low'][i-2]
        return support

    def isResistance(df, i):
        resistance = df['High'][i] > df['High'][i-1] and df['High'][i] > df['High'][i+1] and df['High'][i+1] > df['High'][i+2] and df['High'][i-1] > df['High'][i-2]
        return resistance

    levels = []
    for i in range(2, df.shape[0]-2):
        if isSupport(df, i):
            levels.append((i, df['Low'][i]))
        elif isResistance(df, i):
            levels.append((i, df['High'][i]))

    s = np.mean(df['High'] - df['Low'])

    def isFarFromLevel(l):
        return np.sum([abs(l - x) < s for x in levels]) == 0

    # %%

    levels_filtered = []
    for i in range(2, df.shape[0] - 2):
        if isSupport(df, i):
            l = df['Low'][i]
            if isFarFromLevel(l):
                levels_filtered.append((i, l))
        elif isResistance(df, i):
            l = df['High'][i]
            if isFarFromLevel(l):
                levels_filtered.append((i, l))
    levels_filtered = [[round(x, 2) for x in tup] for tup in levels]
    second_values = []
    for inner_list in levels:
        second_value = inner_list[1]
        second_values.append(second_value)
    second_values.sort()
    return second_values

def plot_all(symbol, start_date):
    df = get_stock_data(symbol, start_date)
    levels = plot_support_resistance_levels(symbol, start_date)

    # Create a plot of the candlestick data using mplfinance
    mpf.plot(df, type='candle', volume=False, style='charles', figratio=(10, 6), datetime_format='%d-%m-%Y',
             hlines=dict(hlines=levels, colors='b'))

    plt.show()



if __name__ == '__main__':

    symbol = 'ITC.NS'
    start_date = '2020-01-01'
    a= plot_support_resistance_levels(symbol, start_date)
    print(a)
    plot_all(symbol, start_date)