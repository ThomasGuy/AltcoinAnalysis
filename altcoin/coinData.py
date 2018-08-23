import logging
import warnings
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .getData import getData

# log = logging.getLogger(__name__)


class Portfolio(list):
    """
    A collection of coins in Portfolio
    """
    pass


class CoinData:
    """
    Generate data frame, set big, small and long ewma's.
    Set daily and cumlative returns, and add coin to Portfolio.
    Check volatility and calculate risk
    """
    portfolio = Portfolio()
    step = '6h'
    sma = 10
    bma = 27
    lma = 74

    @classmethod
    def setParams(cls, step, sma, bma, lma):
        cls.step = step
        cls.sma = sma
        cls.bma = bma
        cls.lma = lma

    def __init__(self, coin):
        self.coin = coin
        self.portfolio.append(self)

    def __repr__(self):
        return self.coin.title()

    def getCoinData(self):
        dataf = self._get_DFTable()
        return (dataf, self._crossover(dataf))

    def plot(self, start=None, finish=None):
        df = self._get_DFTable().loc[start:finish]
        cross = self._crossover(df)
        self._plott(df, cross, title=self.__repr__())

    def _get_DFTable(self):
        """
        Initialize coin DataFrame
        """
        df = getData(self.coin, self.step)
        df.drop_duplicates()
        df = df.groupby('MTS')['open', 'close', 'high', 'low', 'volume'].mean()
        df['sewma'] = df['close'].ewm(span=self.sma).mean()
        df['bewma'] = df['close'].ewm(span=self.bma).mean()
        df['lewma'] = df['close'].ewm(span=self.lma).mean()
        return df

    def _crossover(self, dataset):
        """From DataFrame 'dataset'. Return a DataFrame of all the crossing points
         of the small and medium moving averages"""
        record = []
        # we have equality in the first row of dataset, NOTE start should be chosen at
        # least a week after the dataset starts so the ewma's settle down
        if dataset.iloc[0]['sewma'] != dataset.iloc[0]['bewma']:
            Higher = dataset.iloc[0]['sewma'] > dataset.iloc[0]['bewma']
        else:
            Higher = dataset.iloc[1]['sewma'] > dataset.iloc[1]['bewma']

        with warnings.catch_warnings():                              #  Catch Numpy warning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for date, row in dataset.iterrows():
                if Higher:
                    # Sell condition
                    if row['sewma'] / row['bewma'] < 0.9965:
                        record.append([date, row['close'], 'Sell'])
                        Higher = not Higher
                else:
                    # Buy condition
                    if row['sewma'] / row['bewma'] > 1.0035:
                        record.append([date, row['close'], 'Buy'])
                        Higher = not Higher

        cross = pd.DataFrame(record, columns=('MTS close Transaction').split())
        cross.set_index('MTS', drop=True, inplace=True)
        return cross

    def _plott(self, dataf, trend, title='Title'):
        """
        Plot DataFrame
        """
        fig = plt.figure(figsize=(16, 9))
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(dataf.index, dataf['sewma'], label='sma={}'.format(self.sma), color='blue')
        axes.plot(dataf.index, dataf['bewma'], label='bma={}'.format(self.bma), color='red')
        axes.plot(dataf.index, dataf['lewma'], label='lma={}'.format(self.lma), color='orange', alpha=.5)
        axes.plot(dataf.index, dataf['close'], label='close', color='green', alpha=.5)
        # axes.plot(dataf.index, dataf['high'], label='high', color='pink', alpha=.5)

        # Plot the sewma/bewma crossing buy & sell points
        sold = pd.DataFrame(trend[trend['Transaction'] == 'Sell']['close'])
        axes.scatter(sold.index, sold['close'], color='r', label='Sell', lw=3)
        bought = pd.DataFrame(trend[trend['Transaction'] == 'Buy']['close'])
        axes.scatter(bought.index, bought['close'], color='g', label='Buy', lw=3)

        axes.set_ylabel('Closing Price')
        axes.set_xlabel('Date')
        axes.set_title(title + " - freq:'{}'  Latest Price: ${:.2f}".format(self.step, dataf['close'].iloc[-1]),
                       fontdict={'fontsize': 25})
        axes.grid(color='b', alpha=0.5, linestyle='--', linewidth=0.5)
        axes.grid(True)
        plt.legend()
        plt.show()

    def trendFollower(self, start, finish):
        """
        watch profits as we follow the buy/sell trends
        # NOTE start should be chosen at least a week after the whole
        # dataset starts so the ewma's settle down
        """
        dataf, cross = self.getCoinData()

        data = pd.DataFrame(dataf.loc[start:finish])
        data['return'] = data['close'].pct_change(1)
        data['cumlative'] = (1 + data['return']).cumprod()

        trend = pd.DataFrame(cross.loc[start:finish])
        trend.insert(2, 'Profit', '')
        trend.insert(3, 'Coins', '')

        # Add the first row
        first_close = data['close'].iloc[0]
        if data['sewma'].iloc[0] < data['bewma'].iloc[0]:
            trend.loc[data.index[0]] = [first_close, 'Invest', 1, np.NaN]
        else:
            trend.loc[data.index[0]] = [first_close, 'Invest', 1, 1 / first_close]
        trend.sort_index(inplace=True)
        # Update the trend following rows
        count = 0
        for date, row in trend.iterrows():
            if row[1] == 'Buy':
                profit = trend.iloc[count - 1, 2]
                trend.loc[date, 'Profit'] = profit
                trend.loc[date, 'Coins'] = profit / trend.loc[date, 'close']
            if row[1] == 'Sell':
                coins = trend.iloc[count - 1, 3]
                trend.loc[date, 'Profit'] = coins * trend.loc[date, 'close']
                trend.loc[date, 'Coins'] = np.NaN
            count += 1
        
        # Add the last row
        last_close = data['close'].iloc[-1]
        if trend['Transaction'].iloc[-1] == 'Buy':
            coins = trend['Coins'].iloc[-1]

            trend.loc[data.index[-1]] = [last_close, 'Cash Out', coins * last_close, np.NaN]
        else:
            trend.loc[data.index[-1]] = [last_close, 'Cash out', trend['Profit'].iloc[-1], np.NaN]

        return (trend, data)
