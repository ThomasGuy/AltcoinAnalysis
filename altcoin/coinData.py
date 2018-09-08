import logging
import warnings
from datetime import timedelta, datetime

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num, WeekdayLocator, DayLocator, MONDAY
from mpl_finance import candlestick_ohlc
import numpy as np

from .getData import getData


# log = logging.getLogger(__name__)


class Portfolio(dict):
    """
    A collection of coins in Portfolio
    """
    def removeit(self, coin):
        # just use builtin pop()
        self.pop(coin)

    def volatility(self):
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_axes([0, 0, 1, 1])
        # ax.autoscale_view()
        ax.set_title('Portfolio Volatillity')
        for coin in self.values():
            df, cross = coin.getCoinData()
            df['returns'] = df['close'].pct_change(1)
            df['returns'].plot(kind='kde', label=coin.name)
        plt.legend()
        plt.show()

    def portfolioDF(self, start=None, finish=None):
        """
        A dictionary of Portfolio dataframes
        """
        df_portfolio = {}
        step = CoinData.params['step']
        for key, coin in self.items():
            sf = coin.get_data(key, step)['close'].loc[start:finish]
            # use resample to make sure we have a complete dataset
            DF = pd.DataFrame(sf).resample('6H', loffset='-5H').ffill()[1:].copy()

            df_portfolio[key] = DF
        return df_portfolio
    
    def normed_return(self, start=None, finish=None):
        normed = self.portfolioDF(start, finish)
        for df in normed.values():
            df['Normed Return'] = df['close'] / df.iloc[0]['close']
        return normed

    def stocks(self, start=None, finish=None):
        """
        Return A dataframe of portfolio stock closing price for risk optimization
        """
        column =[]
        data  = []
        for key, df in self.portfolioDF(start, finish).items():
            column.append(key)
            data.append(df)
        stocks = pd.concat(data, axis=1)
        stocks.columns = column
        return stocks


class CoinData:
    """
    Generate data frame, set big, small and long ewma's.
    Set daily and cumlative returns, and add coin to Portfolio.
    Check volatility and calculate risk
    """
    portfolio = Portfolio()
    params = {
        'sma': 10,
        'bma': 27,
        'lma': 74,
        'step': '6h'
    }

    @classmethod
    def setParams(cls, sma, bma, lma, step):
        cls.params['sma'] = sma
        cls.params['bma'] = bma
        cls.params['lma'] = lma
        cls.params['step'] = step

    def __init__(self, name, get_data=getData, **kwargs):
        # self.params = dict(CoinData.default_params)
        # self.params.update(kwargs)
        self.name = name
        self.portfolio[self.name] = self
        self.get_data = get_data

    def __str__(self):
        return self.name.title() 

    def __repr__(self):
        return self.name

    @property
    def value(self):
        return self.get_data(self.name, self.params['step'])['close'].iloc[-1]

    def getCoinData(self):
        dataf = self._init_DF(**self.params)
        return (dataf, self._crossover(dataf))

    def plot(self, start=None, finish=None, title=None):
        title = title or self.__str__().capitalize()
        df, cross = self.getCoinData()
        # start = self._begining(start, df)
        df = df.loc[start:finish]
        cross = cross.loc[start:finish]
        self._plott(df, cross, **self.params, title=title)

    def trendPlot(self, start=None, finish=None):
        self._trendPlott(*self.trendFollower(start, finish), title=self.__str__())

    def _init_DF(self, sma, bma, lma, step):
        """
        Initialize DataFrame
        """
        df = self.get_data(self.name, step)
        df.drop_duplicates()
        df = df.groupby('MTS')['open', 'close', 'high', 'low', 'volume'].mean()
        df['sewma'] = df['close'].ewm(span=sma).mean()
        df['bma'] = df['close'].rolling(bma).mean()
        df['lma'] = df['close'].rolling(lma).mean()
        # Lop off the the first 'bma' number of rows to let ewma's to settle
        # So we don't need to adjust the start time with func. begining()
        return df.iloc[bma:]

    def _crossover(self, dataset):
        """From DataFrame 'dataset'. Return a DataFrame of all the crossing points
         of the small and medium moving averages"""
        record = []
        Higher = dataset.iloc[0]['sewma'] > dataset.iloc[0]['bma']

        #  Catch Numpy warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for date, row in dataset.iterrows():
                if Higher:
                    # Sell condition
                    if row['sewma'] / row['bma'] < 0.9965:
                        record.append([date, row['close'], 'Sell'])
                        Higher = not Higher
                else:
                    # Buy condition
                    if row['sewma'] / row['bma'] > 1.0035:
                        record.append([date, row['close'], 'Buy'])
                        Higher = not Higher

        cross = pd.DataFrame(record, columns=('MTS close Transaction').split())
        cross.set_index('MTS', drop=True, inplace=True)
        return cross

    def _plott(self, dataf, trend, sma, bma, lma, step, title):
        """
        Plot DataFrame
        """
        fig = plt.figure(figsize=(16, 9))
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(dataf.index, dataf['sewma'], label='sewma={}'.format(sma), color='blue')
        axes.plot(dataf.index, dataf['bma'], label='bma={}'.format(bma), color='red')
        axes.plot(dataf.index, dataf['lma'], label='lma={}'.format(lma), color='orange', alpha=.5)
        axes.plot(dataf.index, dataf['close'], label='close', color='green', alpha=.5)
        # axes.plot(dataf.index, dataf['high'], label='high', color='pink', alpha=.5)

        # Plot the sewma/bma crossing buy & sell points
        sold = pd.DataFrame(trend[trend['Transaction'] == 'Sell']['close'])
        axes.scatter(sold.index, sold['close'], color='r', label='Sell', lw=3)
        bought = pd.DataFrame(trend[trend['Transaction'] == 'Buy']['close'])
        axes.scatter(bought.index, bought['close'], color='g', label='Buy', lw=3)

        axes.set_ylabel('Closing Price')
        axes.set_xlabel('Date')
        axes.set_title(title + " - freq:'{}'  Last Price: ${:.2f}".format(step, self.value),
                       fontdict={'fontsize': 25})
        axes.grid(color='b', alpha=0.5, linestyle='--', linewidth=0.5)
        axes.grid(True)
        plt.legend()
        plt.show()

    def trendFollower(self, start=None, finish=None):
        """
        watch profits as we follow the buy/sell trends
        """
        dataf, cross = self.getCoinData()
        # start = self._begining(start, dataf)

        data = pd.DataFrame(dataf.loc[start:finish])
        data['return'] = data['close'].pct_change(1)
        data['cumlative'] = (1 + data['return']).cumprod()

        trend = pd.DataFrame(cross.loc[start:finish])
        trend.insert(2, 'Profit', '')
        trend.insert(3, 'Coins', '')

        # Add the first row
        trend.loc[data.index[0]] = [data['close'].iloc[0], 'Invest', 1, 1 / data['close'].iloc[0]]
        trend.sort_index(inplace=True)
        # Update the trend following rows
        count = 0
        for date, row in trend.iterrows():
            if row[1] == 'Buy':
                profit = trend.iloc[count - 1, 2]
                trend.loc[date, 'Profit'] = profit
                trend.loc[date, 'Coins'] = profit / row[0]
            elif row[1] == 'Sell':
                coins = trend.iloc[count - 1, 3]
                trend.loc[date, 'Profit'] = coins * row[0]
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

    def _trendPlott(self, trend, data, title):
        """
        plot trend following result against cumlative return (HODL)
        """
        fig = plt.figure(figsize=(16, 3))
        axes = fig.add_axes([0, 0, 1, 1])
        axes.plot(trend.index, trend['Profit'], label='Trend')
        axes.plot(data.index, data['cumlative'])
        axes.set_ylabel('Profit')
        axes.set_xlabel('Date')
        axes.set_title(title + " - profit: {:.2f}%".format(trend['Profit'].iloc[-1] * 100),
                       fontdict={'fontsize': 25})
        axes.grid(color='b', alpha=0.5, linestyle='--', linewidth=0.5)
        axes.grid(True)
        plt.legend()
        plt.show()
 
    # def _begining(self, start, dataf):
    #     """
    #     NOTE start should be chosen at least a week after the whole
    #     dataset starts so the ewma's settle down
    #     """
    #     begin = dataf.index.min() + timedelta(7)
    #     # check for start = None
    #     if start:
    #         if begin > datetime.strptime(start, '%Y-%m-%d'):
    #             start = datetime.strftime(begin, "%Y-%m-%d %H:%M:%S")
    #     else:
    #         return datetime.strftime(begin, "%Y-%m-%d %H:%M:%S")
    #     return start

    def plotCandles(self, days=30, width=0.2, title=None):
        self.candles(days, width, title, **self.params)

    def candles(self, days, width, title, sma, bma, lma, step):
        title = title or self.name.capitalize()
        dataf = self._init_DF(sma, bma, lma, step)
        start = (dataf.index[-1] - timedelta(days)).strftime('%Y-%m-%d')
        data = dataf.loc[start:].copy().reset_index()
        data['date_ax'] = data['MTS'].apply(lambda date: date2num(date))
        df_values = [tuple(vals) for vals in data[[
            'date_ax', 'open', 'high', 'low', 'close']].values]
        mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
        alldays = DayLocator()              # minor ticks on the days
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12

        fig, ax = plt.subplots(figsize=(18, 8))
        fig.subplots_adjust(bottom=0.2)
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
        ax.xaxis.set_major_formatter(weekFormatter)
        ax.autoscale_view()
        ax.xaxis.grid(True, 'major')
        ax.grid(True)
        ax.set_facecolor('lightgrey')
        ax.set_title("{} - freq'{}' - latest price: ${:.4f}".
                     format(title, step, self.value), fontsize=20)
        candlestick_ohlc(ax, df_values, width=width, colorup='g', colordown='r')

        # plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.plot(data.date_ax, data['sewma'], label=f'sma={sma}', color='green')
        ax.plot(data.date_ax, data['bma'], label=f'bma={bma}', color='blue')
        ax.plot(data.date_ax, data['lma'], label=f'lma={lma}', color='orange')
        plt.legend()
        plt.show()
