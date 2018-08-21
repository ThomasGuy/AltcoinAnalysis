# -*- coding: utf-8 -*-
"""
Created on Fri Apr  13 16:00:00 2018

@author: Sporty

Here we will keep track of two timelines,
one with no stop loss 'total' and 'dollar'
the other with a stoploass in action with 'stop_total' and 'stop_dollar'.
Thus we can plot both timelines together and optimise for stoploss
"""

import pandas as pd
import matplotlib.pyplot as plt
from .getBitfinexCandles import top_twenty5 as coins
from .getData import getData

# Initial investment
investment = 1000


def init(dataset, colName):
    record, stoploss = _initData(dataset, colName)
    return _trailingStop(dataset, record, stoploss, colName)


def _initData(dataset, colName):
    # We start with $1000
    dollar = investment
    stop_dollar = investment
    # The 'sma' is higher than the 'bma' ?
    if dataset.iloc[0][colName[0]] > dataset.iloc[0][colName[1]]:
        # start with coins
        total = dollar / dataset.iloc[0]['open']  # coins
        stop_total = total
    else:
        # start wiht dollars
        total = dollar
        stop_total = dollar

    # results are a series of lists which will later be DataFramed
    return (
        [[dataset.index[0], dataset.iloc[0]['close'], 'Start', dollar, total]],
        [[dataset.index[0], dataset.iloc[0]['close'], 'Start', stop_dollar, stop_total]]
    )


def _trailingStop(dataset, record, stoploss, colName):
    sewma, bewma = colName
    STOP = False
    stopIt = 1.1  # 10% stop loss
    Higher = dataset.iloc[0][sewma] > dataset.iloc[0][bewma]
    # high to track stoploss condition
    high = dataset.iloc[0]['high']
    # dataset is a pandas dataFrame we itereate row by row
    for date, row in dataset.iterrows():
        if Higher:

            # Sell Condition
            if row[sewma] / row[bewma] < 0.9965:
                # Was Higher now lower so sell coins
                total = record[-1][-1] * row['close']
                dollar = total
                if not STOP:
                    # STOP is False we update like a normal sell condition
                    stop_total = stoploss[-1][-1] * row['close']
                    stop_dollar = stop_total
                else:
                    # leave record the same as we have already stopped
                    stop_total = stoploss[-1][-1]
                    stop_dollar = stoploss[-1][-2]

                record.append([date, row['close'], 'Sell', dollar, total])
                stoploss.append([date, row['close'], 'Sell', stop_dollar,
                                 stop_total])
                # switch over to buy condition
                Higher = not Higher

            # Look to buy in again after a stoploss.
            # If we're still higher with +ve gradient we should buy again
            if STOP and Higher:
                if row['returns'] > 0.02:
                    # now we can buy
                    stop_total = stoploss[-1][-1] / row['close']
                    stop_dollar = stoploss[-1][-2]
                    stoploss.append([date, row['close'], 'Stop_Buy',
                                     stop_dollar, stop_total])
                    STOP = False
                    high = row['close']

            # we are looking when to sell or STOP
            high = max(high, row['close'])
            # STOP condition
            if not STOP and Higher and (high / row['close'] >= stopIt):
                # We must stoploss
                stop_total = stoploss[-1][-1] * row['close']
                stop_dollar = stop_total
                stoploss.append([date, row['close'], 'Stop_Sell', stop_dollar,
                                 stop_total])
                STOP = True

        else:   # Buy condition
            if row[sewma] / row[bewma] > 1.0035:
                # now we can buy
                total = record[-1][-1] / row['close']
                dollar = record[-1][-2]
                stop_total = stoploss[-1][-1] / row['close']
                stop_dollar = stoploss[-1][-2]
                record.append([date, row['close'], 'Buy', dollar, total])
                stoploss.append([date, row['close'], 'Buy', stop_dollar,
                                 stop_total])
                STOP = False
                Higher = not Higher
                high = row['close']

    price = pd.DataFrame(record, columns=('Date Close Transaction Dollar Total').split())
    stop_price = pd.DataFrame(stoploss, columns=('Date', 'Close', 'Transaction', 'Stop_Dollar', 'Stop_Total'))
    price.set_index('Date', drop=True, inplace=True)
    stop_price.set_index('Date', drop=True, inplace=True)
    return (price, stop_price)


def plotDataset(dataset, price, stop_price, text):
    sym, step, sma, bma, title = text
    fig = plt.figure(figsize=(16, 9))
    axes = fig.add_axes([0, 0, 1, 1])
    axes.plot(dataset.index, dataset['sewma'], label='ewma={}'.format(sma),
              color='blue')
    axes.plot(dataset.index, dataset['bewma'], label='ewma={}'.format(bma),
              color='red')
    axes.plot(dataset.index, dataset['close'], label='close', color='green',
              alpha=.5)
    axes.plot(dataset.index, dataset['longewma'], label='longma',
              color='orange', alpha=.5)
    axes.plot(dataset.index, dataset['high'], label='high', color='pink',
              alpha=.5)

    # Plot the sewma/bewma crossing buy & sell points
    sold = pd.DataFrame(price[price['Transaction'] == 'Sell']['Close'])
    axes.scatter(sold.index, sold['Close'], color='r', label='Sell', lw=3)
    bought = pd.DataFrame(price[price['Transaction'] == 'Buy']['Close'])
    axes.scatter(bought.index, bought['Close'], color='g', label='Buy', lw=3)

    # Plot the Stoploss sell & buy points
    stop_sold = pd.DataFrame(stop_price[stop_price['Transaction'] == 'Stop_Sell']['Close'])
    axes.scatter(stop_sold.index, stop_sold['Close'], color='r',
                 label='Stop_Loss - Sell', lw=7, alpha=.4)
    stop_bought = pd.DataFrame(stop_price[stop_price['Transaction'] == 'Stop_Buy']['Close'])
    axes.scatter(stop_bought.index, stop_bought['Close'], color='g',
                 label='Stop_Buy', lw=7, alpha=.4)

    profit = dollarTotal(dataset, price, 'Total')
    stop_profit = dollarTotal(dataset, stop_price, 'Stop_Total')
    axes.set_title('{} {} at {} interval sewma={} bewma={} Dollars= {} Stop_= {}'
                   .format(title, sym, step, sma, bma, profit, stop_profit))
    axes.set_ylabel('closing pric')
    axes.set_xlabel('Date')
    axes.grid(color='b', alpha=0.5, linestyle='--', linewidth=0.5)
    axes.grid(True)
    plt.legend()
    plt.show()


# plot price against stop loss price
def plotDollars(dataset, price, stop_price, text):
    sym, step, sma, bma, title = text
    profit = dollarTotal(dataset, price, 'Total')
    stop_profit = dollarTotal(dataset, stop_price, 'Stop_Total')
    price['Dollar'].plot(label='sma/bma=${}'.format(profit), figsize=(16, 6),
                         title='{} {} sewma:{} bewma:{} at {} intervals'.
                         format(sym, title, sma, bma, step), lw=2)
    stop_price['Stop_Dollar'].plot(label='Stoploss=${}'.format(stop_profit), lw=1)
    dataset['Cumlative Return'] = dataset['Cumlative Return'] * investment
    dataset['Cumlative Return'].plot(label='HODL=${}'.format(int(dataset['Cumlative Return'].iloc[-1])), lw=1)
    plt.grid(color='b', alpha=0.5, linestyle='--', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()


# Dollar total for a dataframe
def dollarTotal(dataset, price, col):
    if price['Transaction'].iloc[-1] in ['Buy', 'Stop_Buy']:
        return int(dataset.iloc[-1]['close'] * price.iloc[-1][col])
    else:
        return int(price.iloc[-1][col])


def showDollar():
    step = '6h'
    colName = ('sewma', 'bewma')
    for coin in coins:
        sym = coin[:3]
        sma = 9
        bma = 24
        text = (sym, step, sma, bma, 'Stoploss Comparision')
        data = pd.read_csv('Data\\MasterData\\{}\\{}_Candles_{}.csv'
                           .format(step, sym, step), index_col='MTS',
                           parse_dates=True)

        data['sewma'] = data['close'].ewm(span=sma).mean()
        data['bewma'] = data['close'].ewm(span=bma).mean()
        data['longewma'] = data['close'].ewm(span=55).mean()
        data['returns'] = data['close'].pct_change(3)
        dataset = data.loc[pd.to_datetime('2017-09-08'):]
        result, stop_result = init(dataset, colName)
        plotDollars(dataset, result, stop_result, text)


def ewma_Plot(sym, step, sewma, bewma, start, end):
    data = getData(sym, step).loc[pd.to_datetime(start): pd.to_datetime(end)]
    text = (sym, step, sewma, bewma, 'Simple Plot')
    colName = ('sewma', 'bewma')
    data['sewma'] = data['close'].ewm(span=sewma).mean()
    data['bewma'] = data['close'].ewm(span=bewma).mean()
    data['longewma'] = data['close'].ewm(span=55).mean()
    data['returns'] = data['close'].pct_change(2)
    data['ret'] = data['close'].pct_change(1)
    # Add on HODL or cumlative return for just holding
    data['Cumlative Return'] = (1 + data['ret']).cumprod()
    result, stop_result = init(data, colName)
    plotDataset(data, result, stop_result, text)
    plotDollars(data, result, stop_result, text)

    return (result, stop_result)
